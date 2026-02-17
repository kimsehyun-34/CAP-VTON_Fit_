import gc
import numpy as np
from PIL import Image
from capvton.transform import LeffaTransform
from capvton.model import LeffaModel
from capvton.inference import LeffaInference
from capvton_utils.garment_agnostic_mask_predictor import AutoMasker
from capvton_utils.densepose_predictor import DensePosePredictor
from capvton_utils.utils import resize_and_center
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import cv2
import os


class CAPVirtualTryOn:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir

        # --- Lightweight preprocessing models ---
        self.mask_predictor = AutoMasker(
            densepose_path=f"{ckpt_dir}/densepose",
            schp_path=f"{ckpt_dir}/schp",
        )

        # Reuse detectron2 predictor from AutoMasker (~170MB GPU saved)
        self.densepose_predictor = DensePosePredictor(
            predictor=self.mask_predictor.densepose_processor.predictor,
        )

        # Human parsing: ONNX on CPU — no GPU cost
        self.parsing = Parsing(
            atr_path=f"{ckpt_dir}/humanparsing/parsing_atr.onnx",
            lip_path=f"{ckpt_dir}/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path=f"{ckpt_dir}/openpose/body_pose_model.pth",
        )

        # --- Heavy diffusion models: lazy-loaded, only one on GPU at a time ---
        self._vt_inference_hd = None
        self._vt_inference_dc = None
        self._skin_pipe = None

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------
    def _free_gpu(self):
        """Remove heavy diffusion models from GPU."""
        # float16 diffusers pipelines don't release CUDA memory properly
        # with .to("cpu") — delete them entirely instead
        if self._skin_pipe is not None:
            del self._skin_pipe
            self._skin_pipe = None
        # nn.Module VT models can safely move to CPU
        if self._vt_inference_hd is not None:
            self._vt_inference_hd.model.cpu()
        if self._vt_inference_dc is not None:
            self._vt_inference_dc.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_preprocessing(self):
        """Move all preprocessing GPU models to CPU (~880MB freed)."""
        for accessor in [
            lambda: self.mask_predictor.schp_processor_atr.model,
            lambda: self.mask_predictor.schp_processor_lip.model,
            lambda: self.mask_predictor.densepose_processor.predictor.model,
            lambda: self.openpose.preprocessor.body_estimation.model,
        ]:
            try:
                accessor().cpu()
            except (AttributeError, RuntimeError):
                pass
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_preprocessing_on_gpu(self):
        """Reload preprocessing models to GPU (reverses _offload_preprocessing)."""
        for accessor in [
            lambda: self.mask_predictor.schp_processor_atr.model,
            lambda: self.mask_predictor.schp_processor_lip.model,
            lambda: self.mask_predictor.densepose_processor.predictor.model,
            lambda: self.openpose.preprocessor.body_estimation.model,
        ]:
            try:
                accessor().cuda()
            except (AttributeError, RuntimeError):
                pass

    def _get_skin_pipe(self):
        """Lazy-load skin inpainting pipeline onto GPU."""
        self._free_gpu()
        if self._skin_pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16,
            )
            self._skin_pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                f"{self.ckpt_dir}/majicmixRealistic_v7.safetensors",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
        self._skin_pipe = self._skin_pipe.to("cuda")
        return self._skin_pipe

    def _get_vt_inference(self, model_type: str):
        """Lazy-load virtual try-on model onto GPU."""
        self._free_gpu()
        if model_type == "viton_hd":
            if self._vt_inference_hd is None:
                vt_model = LeffaModel(
                    pretrained_model_name_or_path=f"{self.ckpt_dir}/stable-diffusion-inpainting",
                    pretrained_model=f"{self.ckpt_dir}/virtual_tryon.pth",
                    dtype="float16",
                )
                self._vt_inference_hd = LeffaInference(model=vt_model, auto_device=False)
            self._vt_inference_hd.model.to("cuda")
            return self._vt_inference_hd
        else:
            if self._vt_inference_dc is None:
                vt_model = LeffaModel(
                    pretrained_model_name_or_path=f"{self.ckpt_dir}/stable-diffusion-inpainting",
                    pretrained_model=f"{self.ckpt_dir}/virtual_tryon_dc.pth",
                    dtype="float16",
                )
                self._vt_inference_dc = LeffaInference(model=vt_model, auto_device=False)
            self._vt_inference_dc.model.to("cuda")
            return self._vt_inference_dc

    # ------------------------------------------------------------------
    # Skin inpainting (Stage 1)
    # ------------------------------------------------------------------
    def generate_skin(
        self,
        src_image: Image.Image,
        inpaint_mask_img: Image.Image,
        step: int = 20,
        seed: int = 42,
    ) -> Image.Image:
        """Inpaint realistic skin in the masked area."""
        skin_prompt = (
            "Wearing Held Tight Short Sleeve Shirt, high quality skin, realistic, high quality"
        )
        negative_prompt = (
            "Blurry, low quality, artifacts, deformed, ugly, texture, "
            "watermark, text, bad anatomy, extra limbs, face, hands, fingers"
        )

        # Run OpenPose while preprocessing is still on GPU
        openpose_result = self.openpose(src_image)
        openpose_image = (
            openpose_result.get("image") if isinstance(openpose_result, dict) else openpose_result
        )
        if not isinstance(openpose_image, Image.Image):
            raise TypeError(f"Invalid OpenPose output: {type(openpose_image)}")

        # Offload ALL preprocessing to CPU to make room for skin_pipe (~3.5GB)
        self._offload_preprocessing()

        generator = torch.Generator(device="cuda").manual_seed(seed)
        skin_pipe = self._get_skin_pipe()

        generated_image = skin_pipe(
            prompt=skin_prompt,
            negative_prompt=negative_prompt,
            image=src_image,
            mask_image=inpaint_mask_img,
            control_image=openpose_image,
            width=src_image.width,
            height=src_image.height,
            num_inference_steps=step,
            generator=generator,
            guidance_scale=7.0,
        ).images[0]

        # Composite: keep original outside mask, inpainted inside
        src_np = np.array(src_image)
        mask_np = np.array(inpaint_mask_img.convert("L"), dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, np.newaxis]
        generated_np = np.array(generated_image)
        final_np = src_np * (1.0 - mask_np) + generated_np * mask_np
        return Image.fromarray(final_np.astype(np.uint8))

    # ------------------------------------------------------------------
    # Main prediction entry-point
    # ------------------------------------------------------------------
    def capvton_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        output_path: str = None,
        step=20,
        cross_attention_kwargs={"scale": 3},
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        src_mask_path=None,
    ):
        assert control_type in [
            "virtual_tryon",
            "pose_transfer",
        ], f"Invalid control type: {control_type}"

        # Ensure preprocessing models are on GPU (may have been offloaded)
        self._ensure_preprocessing_on_gpu()

        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        # ---- Step 1: Parsing (CPU ONNX — no GPU cost) ----
        parsing_map, _ = self.parsing(src_image.resize((768, 1024)))
        parsing_map = np.array(parsing_map)

        upper_body_mask_np = np.isin(parsing_map, [4]).astype(np.uint8)
        arms_mask_np = np.isin(parsing_map, [14, 15]).astype(np.uint8)
        hands_mask_np = np.isin(parsing_map, [20, 21]).astype(np.uint8)
        inpaint_mask_np = upper_body_mask_np | arms_mask_np | hands_mask_np

        kernel = np.ones((10, 10), np.uint8)
        inpaint_mask_np_dilated = cv2.dilate(
            inpaint_mask_np.astype(np.uint8), kernel, iterations=1
        )
        inpaint_mask_img = Image.fromarray(inpaint_mask_np_dilated * 255)

        # Early exit: no body area detected
        if not np.any(inpaint_mask_np):
            empty_mask = Image.fromarray(np.zeros((1024, 768), dtype=np.uint8))
            if output_path:
                src_image.save(output_path)
            return src_image, empty_mask, empty_mask, src_image

        # ---- Step 2: Skin inpainting ----
        # (internally: runs OpenPose → offloads preprocessing → loads skin_pipe)
        agnostic_image = self.generate_skin(
            src_image=src_image,
            inpaint_mask_img=inpaint_mask_img,
            step=step,
            seed=seed,
        )

        # ---- Step 3: Garment mask + DensePose ----
        # Delete skin_pipe, reload preprocessing for masking
        self._free_gpu()
        self._ensure_preprocessing_on_gpu()

        if control_type == "virtual_tryon":
            garment_mapping = {
                "dresses": "overall",
                "upper_body": "upper",
                "lower_body": "lower",
                "short_sleeve": "short_sleeve",
                "shorts": "shorts",
            }
            garment_type_hd = garment_mapping.get(vt_garment_type, "upper")
            mask = self.mask_predictor(agnostic_image, garment_type_hd)["mask"]
            if src_mask_path:
                mask.save(src_mask_path)
        else:
            agnostic_np = np.array(agnostic_image)
            mask = Image.fromarray(np.ones_like(agnostic_np, dtype=np.uint8) * 255)

        agnostic_np = np.array(agnostic_image)
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(agnostic_np)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(agnostic_np)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)

        # ---- Step 4: Virtual try-on inference ----
        # Offload preprocessing (~880MB) → load VT model (~3.5GB)
        self._offload_preprocessing()

        transform = LeffaTransform()
        data = {
            "src_image": [agnostic_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self._get_vt_inference(vt_model_type)

        garment_prompt = "High quality skin, lifelike details, realistic textures, full masking range"
        negative_prompt = "distorted, blurry, low quality, artifact, background, clothes"

        result = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            cross_attention_kwargs=cross_attention_kwargs,
            seed=seed,
            repaint=vt_repaint,
            prompt=garment_prompt,
            negative_prompt=negative_prompt,
        )

        gen_image = result["generated_image"][0]

        if output_path:
            gen_image.save(output_path)
            gt_path = os.path.join(os.path.dirname(output_path), "ground_truth.png")
            pred_path = os.path.join(os.path.dirname(output_path), "prediction.png")
            ref_image.save(gt_path)
            gen_image.save(pred_path)
            print(f"Ground Truth saved at: {gt_path}")
            print(f"Prediction saved at: {pred_path}")

        return gen_image, mask, densepose, agnostic_image
