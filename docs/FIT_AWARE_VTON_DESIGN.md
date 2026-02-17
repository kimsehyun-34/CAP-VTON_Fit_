# Fit-Aware Virtual Try-On System Design Document

> CaP-VTON 기반 체형/치수 반영 가상피팅 업그레이드 설계서  
> 작성일: 2026-02-17

---

## 목차

1. [시스템 아키텍처 개요](#1-시스템-아키텍처-개요)
2. [카테고리별 측정치 스키마](#2-카테고리별-측정치-스키마)
3. [Fit Predictor 설계](#3-fit-predictor-설계)
4. [Fit-aware Layout Generator](#4-fit-aware-layout-generator)
5. [CaP-VTON 코드베이스 수정 계획](#5-cap-vton-코드베이스-수정-계획)
6. [평가 지표 및 실험 설계](#6-평가-지표-및-실험-설계)
7. [3단계 로드맵](#7-3단계-로드맵)

---

## 1. 시스템 아키텍처 개요

### 1.1 전체 모듈 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Fit-Aware VTON Pipeline                      │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐           │
│  │ Body     │    │ Fit          │    │ Fit-aware Layout│           │
│  │ Estimator│───▶│ Predictor    │───▶│ Generator       │           │
│  │          │    │ (판단/추천)    │    │ (실루엣 생성)     │           │
│  └──────────┘    └──────┬───────┘    └────────┬────────┘           │
│       │                 │                      │                    │
│       │          ┌──────▼───────┐    ┌────────▼────────┐           │
│       │          │ Fit Report   │    │ Layout Cond.    │           │
│       │          │ (텍스트/JSON) │    │ (마스크+SDF)     │           │
│       │          └──────────────┘    └────────┬────────┘           │
│       │                                       │                    │
│  ┌────▼────────────────────────────────────────▼──────────────┐    │
│  │              CaP-VTON Diffusion Pipeline                   │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │    │
│  │  │Skin     │  │Garment   │  │Ref UNet  │  │Gen UNet   │  │    │
│  │  │Inpaint  │  │Agnostic  │  │(의류특징) │  │+FiLM Cond │  │    │
│  │  │(기존)    │  │Mask(기존) │  │(기존)     │  │(확장)      │  │    │
│  │  └─────────┘  └──────────┘  └──────────┘  └───────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 데이터 흐름 (추론 시)

```
입력:
  - 사용자 사진 (768×1024 RGB)
  - 의류 사진 (768×1024 RGB)
  - 사용자 신체 치수 u = {gender, height, chest, waist, hip, ...}
  - 의류 실측 스펙 g = {category, length, chest_width, waist_width, ...}
  - 요청 사이즈: S/M/L/XL

Step 1: Body Estimator (기존 전처리 활용)
  ├── DensePose → 체형 세그먼트 (I map)
  ├── Human Parsing → ATR/LIP 파싱맵
  ├── OpenPose → 骨格 keypoints
  └── (선택) 키포인트로부터 추정 치수 보정

Step 2: Fit Predictor
  ├── 입력: u, g → ease f = normalize(g - u)
  ├── 출력: FitReport {
  │     overall_score: float,          # 0~1 적합도
  │     size_recommendation: str,      # "M 추천"
  │     body_part_tightness: {         # 부위별 -1(타이트)~+1(오버)
  │       chest: 0.3, waist: -0.1, hip: 0.2,
  │       shoulder: 0.0, sleeve: 0.4, length: 0.1
  │     },
  │     risk_parts: ["chest"],         # 리스크 부위
  │     fit_class: {chest: "loose", waist: "regular", ...}
  │   }
  └── 출력: fit_embedding (128-d vector) → Layout Generator로 전달

Step 3: Fit-aware Layout Generator
  ├── 입력: agnostic_mask, densepose, fit_embedding, garment_category
  ├── 출력:
  │     - target_mask: (1, H, W)  거리 기반 soft 마스크
  │     - sdf_map: (1, H, W)     부호 거리 함수 (의류 경계까지)
  │     - hem_lines: (K, 2)      구조선 keypoints (optional)
  └── 학습: pseudo-fit 데이터 증강으로 사전 학습

Step 4: Skin Inpainting (기존 CaP-VTON)
  └── 의류 마스크 영역 피부 인페인팅

Step 5: Diffusion Try-on (CaP-VTON + FiLM 조건 주입)
  ├── Reference UNet: 의류 특징 추출 (기존)
  ├── Generative UNet: 기존 12-ch 입력 + layout_cond (SDF 2ch 추가 = 14ch)
  │     + FiLM 레이어: fit_embedding → scale/shift per block
  └── 출력: 핏 반영 try-on 이미지 (768×1024 RGB)
```

### 1.3 텐서 정의 (상세)

| 텐서 | Shape | dtype | 설명 |
|------|-------|-------|------|
| `user_measurements` | `(B, N_u)` | float32 | 사용자 치수 벡터 (정규화) |
| `garment_measurements` | `(B, N_g)` | float32 | 의류 실측 벡터 (정규화) |
| `ease_vector` | `(B, N_f)` | float32 | 부위별 여유분 (정규화) |
| `fit_embedding` | `(B, 128)` | float32 | Fit Predictor MLP 출력 |
| `tightness_scores` | `(B, K)` | float32 | 부위별 타이트니스 연속값 |
| `fit_class_logits` | `(B, K, 3)` | float32 | 부위별 3-class 로짓 |
| `target_mask` | `(B, 1, H, W)` | float32 | 핏 반영 의류 영역 soft 마스크 |
| `sdf_map` | `(B, 1, H, W)` | float32 | 부호 거리 함수 |
| `layout_cond` | `(B, 2, H/8, W/8)` | float16 | latent space 레이아웃 조건 |
| `film_scale` | `(B, C_block)` | float16 | FiLM scale per U-Net block |
| `film_shift` | `(B, C_block)` | float16 | FiLM shift per U-Net block |

여기서:
- `N_u = 12` (사용자 치수 차원), `N_g = 15` (의류 실측 차원), `N_f = 10` (ease 차원)
- `K = 6` (부위 수: chest, waist, hip, shoulder, sleeve, length)
- `H = 1024, W = 768` (이미지 해상도)
- `C_block ∈ {320, 640, 1280, 1280}` (U-Net 블록별 채널)

### 1.4 학습 단계

**Stage 1: Fit Predictor 학습**
- 룰 기반 MVP 즉시 배포 + 학습 기반 모델 병행 훈련
- 데이터: 공개 사이즈 차트 + 사용자 피드백(추후)

**Stage 2: Layout Generator 학습**
- DressCode/VITON-HD 데이터 + pseudo-fit 증강
- 목표: (agnostic_mask, densepose, fit_emb) → (target_mask, sdf_map)
- 손실: BCE(mask) + L1(SDF) + boundary consistency

**Stage 3: Fit-aware Diffusion Fine-tuning**
- FiLM 레이어만 학습 (기존 가중치 동결)
- 데이터: Stage 2와 동일 + layout GT를 조건으로
- 손실: reconstruction L1 + perceptual + adversarial(optional)

---

## 2. 카테고리별 측정치 스키마

### 2.1 사용자 신체 측정치 (User Body Measurements)

| 항목 | 필드명 | 단위 | 필수 | 설명 |
|------|--------|------|------|------|
| 성별 | `gender` | enum | **필수** | male / female / unisex |
| 키 | `height` | cm | **필수** | |
| 가슴둘레 | `chest` | cm | **필수** | |
| 허리둘레 | `waist` | cm | **필수** | |
| 엉덩이둘레 | `hip` | cm | **필수** | |
| 어깨너비 | `shoulder_width` | cm | 권장 | 미입력 시 키/가슴에서 추정 |
| 팔길이 | `arm_length` | cm | 권장 | 미입력 시 키에서 추정 |
| 인심(다리안쪽) | `inseam` | cm | 권장(하의) | 미입력 시 키에서 추정 |
| 허벅지둘레 | `thigh` | cm | 선택 | |
| 목둘레 | `neck` | cm | 선택 | |
| 체중 | `weight` | kg | 선택 | BMI 기반 보정용 |
| 연령대 | `age_group` | enum | 선택 | 체형 프로파일 보정 |

**추정 공식 (미입력 항목):**
```
shoulder_width ≈ height × 0.259 (남) / 0.243 (여) + chest 보정
arm_length    ≈ height × 0.327 (남) / 0.317 (여)
inseam        ≈ height × 0.45
thigh         ≈ hip × 0.62
```
> 추정치는 `uncertainty` 플래그와 함께 전달 (σ ≈ ±2~3cm)

### 2.2 의류 실측 스펙 (Garment Measurements)

#### 2.2.1 상의 (Upper Body)

| 항목 | 필드명 | 단위 | 필수 | 설명 |
|------|--------|------|------|------|
| 카테고리 | `category` | enum | **필수** | tshirt / shirt / jacket / sweater / ... |
| 총기장 | `length` | cm | **필수** | 뒷중심 기장 |
| 가슴단면 | `chest_width` | cm | **필수** | 가슴 높이 좌우 단면(×2 = 둘레) |
| 어깨너비 | `shoulder` | cm | **필수** | |
| 소매기장 | `sleeve_length` | cm | **필수** | |
| 허리단면 | `waist_width` | cm | 권장 | |
| 밑단단면 | `hem_width` | cm | 권장 | |
| 소매통 | `sleeve_width` | cm | 선택 | 소매부리 단면 |
| 소매산 | `armhole_depth` | cm | 선택 | |

#### 2.2.2 하의 (Lower Body)

| 항목 | 필드명 | 단위 | 필수 | 설명 |
|------|--------|------|------|------|
| 카테고리 | `category` | enum | **필수** | pants / jeans / skirt / shorts |
| 총기장 | `length` | cm | **필수** | |
| 허리단면 | `waist_width` | cm | **필수** | |
| 엉덩이단면 | `hip_width` | cm | **필수** | |
| 허벅지단면 | `thigh_width` | cm | 권장 | |
| 밑위 | `rise` | cm | 권장 | |
| 인심 | `inseam` | cm | 권장 | |
| 밑단단면 | `hem_width` | cm | 선택 | |

#### 2.2.3 원피스 (Dresses/Overall)

| 항목 | 필드명 | 단위 | 필수 | 설명 |
|------|--------|------|------|------|
| 카테고리 | `category` | enum | **필수** | dress / jumpsuit |
| 총기장 | `length` | cm | **필수** | |
| 가슴단면 | `chest_width` | cm | **필수** | |
| 허리단면 | `waist_width` | cm | **필수** | |
| 엉덩이단면 | `hip_width` | cm | 권장 | |
| 어깨너비 | `shoulder` | cm | 권장 | |
| 소매기장 | `sleeve_length` | cm | 권장(소매 있을 때) | |
| 밑단폭 | `hem_width` | cm | 선택 | |

### 2.3 Ease 계산 (부위별 여유분)

```python
# 단면→둘레 변환 (의류 스펙이 단면인 경우)
garment_circumference = garment_width × 2

# ease = (의류 둘레 - 신체 둘레) / 신체 둘레
ease_ratio = (garment_circ - body_circ) / body_circ

# 정규화: [-1, +1] 범위로 클리핑
# -1 = 매우 타이트 (< -15%), +1 = 매우 오버 (> +25%)
normalized_ease = clip((ease_ratio - 0.05) / 0.20, -1, 1)
```

부위별 ease 벡터 `f = [f_chest, f_waist, f_hip, f_shoulder, f_sleeve_len, f_length, f_thigh, ...]`

---

## 3. Fit Predictor 설계

### 3.1 방식 A: 규칙/통계 기반 MVP

**즉시 구현 가능, 학습 데이터 불필요**

#### 입력 정규화
```python
# z-score 정규화 (성별·카테고리별 통계)
u_norm = (u_raw - μ_body[gender]) / σ_body[gender]
g_norm = (g_raw - μ_garment[category]) / σ_garment[category]
```

#### 핵심 로직
```python
class RuleBasedFitPredictor:
    """
    부위별 ease 임계값 기반 판정.
    ease = (garment_circ - body_circ) / body_circ
    """
    # 카테고리별 이상적 ease 범위 (의류공학 표준)
    EASE_STANDARDS = {
        "tshirt": {
            "chest":  {"tight": (-0.05, 0.03), "regular": (0.03, 0.12), "loose": (0.12, 0.30)},
            "waist":  {"tight": (-0.05, 0.05), "regular": (0.05, 0.15), "loose": (0.15, 0.35)},
            "shoulder": {"tight": (-0.02, 0.01), "regular": (0.01, 0.05), "loose": (0.05, 0.12)},
            "length": {"short": (-0.10, 0.0), "regular": (0.0, 0.08), "long": (0.08, 0.20)},
        },
        # ... 카테고리별 확장
    }

    def predict(self, user, garment, size) -> FitReport:
        ease = self._compute_ease(user, garment)
        tightness = {}
        fit_class = {}
        risks = []
        for part, e in ease.items():
            standards = self.EASE_STANDARDS[garment.category][part]
            if e < standards["tight"][0]:
                tightness[part] = -1.0  # 매우 타이트
                fit_class[part] = "too_tight"
                risks.append(part)
            elif standards["tight"][0] <= e < standards["tight"][1]:
                tightness[part] = -0.5
                fit_class[part] = "tight"
            elif standards["regular"][0] <= e < standards["regular"][1]:
                tightness[part] = 0.0
                fit_class[part] = "regular"
            elif standards["loose"][0] <= e < standards["loose"][1]:
                tightness[part] = 0.5
                fit_class[part] = "loose"
            else:
                tightness[part] = 1.0
                fit_class[part] = "too_loose"
                risks.append(part)

        overall = 1.0 - (len(risks) / len(ease))
        return FitReport(overall_score=overall, tightness=tightness,
                        fit_class=fit_class, risk_parts=risks, ...)
```

#### 사이즈 추천
```python
# 모든 사이즈(S/M/L/XL)에 대해 predict → overall_score 최대인 사이즈 추천
# 리스크 부위 0이면서 tightness가 사용자 선호(정핏/오버)에 가장 가까운 것
```

#### 출력 해석
- `risk_parts`: 빨간색으로 UI 표시
- `tightness` 값: 히트맵/바 차트로 시각화
- 불확실성: 추정치수일 경우 `confidence_interval = ±σ` 범위로 판정 변동 표시

---

### 3.2 방식 B: 학습 기반 (회귀+분류, 부위별 멀티헤드)

#### 아키텍처

```
입력: u(12-d) ⊕ g(15-d) ⊕ f(10-d) ⊕ category_emb(16-d)
                    ↓
             SharedEncoder (MLP)
         [53-d] → 256 → 256 → 128
                    ↓
            fit_embedding (128-d)
                    ↓
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  Regression    Classification  Overall
  Head          Head            Head
  (128→64→K)    (128→64→K×3)   (128→1)
        ↓           ↓           ↓
  tightness     fit_class      overall_score
  (K floats)    (K×3 logits)   (1 float, σ)
```

#### 입력 정규화
```python
# 1) 신체 치수: z-score (성별별 평균/표준편차)
u_norm = (u - μ_body[gender]) / σ_body[gender]

# 2) 의류 치수: z-score (카테고리별)
g_norm = (g - μ_garment[cat]) / σ_garment[cat]

# 3) ease: 이미 비율값이므로 [-1, 1] clipping만
f_norm = clip(f, -1, 1)

# 4) category: learnable embedding (16-d)
cat_emb = CategoryEmbedding(num_categories)(cat_id)

# 5) 불확실성 마스크: 추정 항목은 별도 binary flag 추가
uncertainty_mask = [0 if measured else 1 for field in u]
```

#### 손실 함수
```python
# Multi-task Loss
L_total = λ_reg × L_regression + λ_cls × L_classification + λ_ovr × L_overall

# L_regression: 부위별 tightness smooth-L1
L_regression = Σ_k SmoothL1(tightness_pred[k], tightness_gt[k])

# L_classification: 부위별 cross-entropy (tight/regular/loose)
L_classification = Σ_k CrossEntropy(logits[k], class_gt[k])

# L_overall: BCE (binary: 적합/부적합) 또는 MSE (연속 점수)
L_overall = BCE(overall_pred, overall_gt)

# Uncertainty-aware weighting: 추정치수 부위는 손실 가중치 ↓
weight[k] = 1.0 if measured[k] else 0.5
```

#### 출력 해석 (리스크 부위 표시)
```python
# 부위별 리스크 판정
for part_k in range(K):
    prob = softmax(fit_class_logits[k])  # [p_tight, p_regular, p_loose]
    if prob[0] > 0.6 and tightness[k] < -0.5:
        risk[k] = "too_tight"  # 🔴
    elif prob[2] > 0.6 and tightness[k] > 0.5:
        risk[k] = "too_loose"  # 🟡
    else:
        risk[k] = "ok"  # 🟢

# 불확실성: MC Dropout (학습 시 Dropout 유지)
# → 여러번 forward → tightness의 mean, std 계산
# std > threshold면 "불확실" 경고 표시
```

---

## 4. Fit-aware Layout Generator

### 4.1 출력 조합 선택: 마스크 + SDF (비용 대비 최적)

| 출력 | 비용 | 효과 | 채택 |
|------|------|------|------|
| Target Mask만 | 낮음 | 실루엣 범위 제어 가능하나 경계 부드러움 부족 | △ |
| **Mask + SDF** | **중간** | **경계 품질↑ + 타이트/오버 그라디언트 표현** | **✅ 채택** |
| Mask + SDF + Lines | 높음 | 구조선 정밀도↑이나 학습 복잡도 대비 효과 한계 | Phase 3 |

**SDF(Signed Distance Function)의 장점:**
- 의류 경계로부터의 거리를 연속값으로 인코딩 → 타이트(SDF≈0)과 오버(SDF≫0)을 자연스럽게 표현
- 경계 부근에서 부드러운 그라디언트 → 확산 모델의 조건으로 쓸 때 아티팩트 감소
- 부위별 SDF 채널 분리 가능 (향후 확장)

### 4.2 아키텍처

```
입력:
  agnostic_mask:  (B, 1, 256, 192)   # 기존 마스크 다운스케일
  densepose_seg:  (B, 3, 256, 192)   # DensePose I-map RGB
  fit_embedding:  (B, 128)           # Fit Predictor에서

인코더 (경량 U-Net 또는 ConvNext):
  [4ch → 64 → 128 → 256] (3-level)
  + FiLM conditioning at each level

디코더:
  [256 → 128 → 64 → 2ch]
  output_mask: (B, 1, 256, 192) → sigmoid
  output_sdf:  (B, 1, 256, 192) → tanh × max_dist

최종 출력 (원본 해상도):
  bilinear upsample → (B, 2, 1024, 768)
```

### 4.3 FiLM 조건 주입 (핵심 메커니즘)

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_fc = nn.Linear(cond_dim, feature_dim)
        self.shift_fc = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        # x: (B, C, H, W), cond: (B, cond_dim)
        scale = self.scale_fc(cond).unsqueeze(-1).unsqueeze(-1) + 1.0  # 1-centered
        shift = self.shift_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift
```

**주입 지점:**
- Layout Generator 내부: 인코더/디코더 각 레벨의 ResBlock 직후
- Diffusion U-Net 내부: down_block/mid_block/up_block의 ResNet 출력 직후 (새로운 FiLM 어댑터)

### 4.4 Diffusion U-Net 내 레이아웃 조건 주입

기존 CaP-VTON의 Generative UNet은 `12-ch` 입력:
```
[noisy_latent(4) | mask(1) | masked_image(4) | densepose(3)] = 12ch
```

**확장 방안 (2가지 병행):**

**(A) 입력 채널 확장 (+2ch)**
```
[noisy_latent(4) | mask(1) | masked_image(4) | densepose(3) | layout_mask(1) | sdf(1)] = 14ch
```
- `conv_in` 가중치를 14ch로 확장 (기존 12ch 가중치 보존)
- latent space에서: 원본 해상도 → ÷8 다운스케일하여 latent과 동일 spatial 크기

**(B) FiLM 어댑터 (멀티스케일)**
```python
# 각 U-Net 블록에 FiLM 어댑터 추가
class FitFiLMAdapter(nn.Module):
    def __init__(self, fit_embed_dim=128, block_channels=[320, 640, 1280, 1280]):
        self.films = nn.ModuleList([
            FiLMLayer(fit_embed_dim, ch) for ch in block_channels
        ])

    def modulate(self, block_idx, hidden_states, fit_embedding):
        return self.films[block_idx](hidden_states, fit_embedding)
```

**주입 위치 (Generative UNet `forward()` 내):**
```python
# down_blocks 루프 내:
for i, (block, sample) in enumerate(zip(self.down_blocks, down_block_res_samples)):
    # ... 기존 로직 ...
    if fit_film_adapter is not None:
        sample = fit_film_adapter.modulate(i, sample, fit_embedding)

# mid_block 직후:
sample = self.mid_block(sample, ...)
if fit_film_adapter is not None:
    sample = fit_film_adapter.modulate_mid(sample, fit_embedding)

# up_blocks 루프 내:
for i, block in enumerate(self.up_blocks):
    # ... 기존 로직 ...
    if fit_film_adapter is not None:
        sample = fit_film_adapter.modulate_up(i, sample, fit_embedding)
```

### 4.5 Pseudo-Fit 데이터 증강

기존 데이터셋(DressCode 등)에 다중 사이즈 GT가 없으므로, 합성:

```python
def generate_pseudo_fit_layout(original_mask, densepose, target_ease):
    """
    부위별로 다른 팽창/수축을 적용하여 pseudo GT 생성.
    
    Args:
        original_mask: 원본 의류 마스크 (H, W)
        densepose: DensePose I-map (body part별 영역)
        target_ease: {chest: 0.1, waist: -0.05, hip: 0.15, length: 0.05}
    
    Returns:
        deformed_mask, sdf_map
    """
    # 1) DensePose로 부위별 영역 분리
    chest_region = (densepose_I == CHEST_ID)
    waist_region = (densepose_I == WAIST_ID)
    hip_region   = (densepose_I == HIP_ID)
    
    # 2) 부위별 다른 커널 크기로 dilate/erode
    for region, ease in [(chest_region, target_ease["chest"]), ...]:
        kernel_size = int(abs(ease) * MAX_KERNEL)
        if ease > 0:  # 오버핏 → dilate
            region_mask = cv2.dilate(mask * region, kernel(kernel_size))
        else:  # 타이트 → erode
            region_mask = cv2.erode(mask * region, kernel(kernel_size))
        deformed_mask = blend(deformed_mask, region_mask, region)
    
    # 3) 기장 방향 조절 (상/하단 이동)
    if "length" in target_ease:
        shift = int(target_ease["length"] * MAX_SHIFT_PX)
        deformed_mask = shift_mask_vertically(deformed_mask, shift)
    
    # 4) SDF 계산
    sdf_map = compute_sdf(deformed_mask)
    
    # 5) 자연스러운 범위 제한
    deformed_mask = apply_convex_hull_constraint(deformed_mask)
    
    return deformed_mask, sdf_map
```

---

## 5. CaP-VTON 코드베이스 수정 계획

### 5.1 신규 파일 추가

```
capvton/
├── fit/
│   ├── __init__.py
│   ├── schema.py               # 측정치 스키마 (Pydantic 모델)
│   ├── body_estimator.py       # 키포인트→추정치수 변환
│   ├── fit_predictor_rule.py   # 규칙 기반 Fit Predictor (MVP)
│   ├── fit_predictor_ml.py     # 학습 기반 Fit Predictor
│   ├── layout_generator.py     # Fit-aware Layout Generator
│   ├── film_adapter.py         # FiLM 조건 주입 모듈
│   ├── pseudo_augment.py       # Pseudo-fit 데이터 증강
│   └── metrics.py              # 평가 지표
```

### 5.2 기존 파일 수정

#### (1) `capvton/model.py` — LeffaModel 확장

```python
# 변경: new_in_channels 기본값 12 → 14 (layout 조건 2ch 추가)
# 추가: FitFiLMAdapter 초기화 및 forward 전달

class LeffaModel(nn.Module):
    def __init__(self, ..., new_in_channels=14, fit_embed_dim=128):
        ...
        # 기존 모델 로드 후
        self.fit_film_adapter = FitFiLMAdapter(
            fit_embed_dim=fit_embed_dim,
            block_channels=list(self.unet.config.block_out_channels),
        )
```

#### (2) `capvton/diffusion_model/unet_gen.py` — forward에 fit 조건 전달

```python
def forward(self, ..., 
            reference_features=None,
            fit_embedding=None,        # 추가
            fit_film_adapter=None,     # 추가
):
    ...
    # down_blocks 루프 내, 각 블록 출력 직후:
    if fit_film_adapter is not None and fit_embedding is not None:
        sample = fit_film_adapter.modulate(block_idx, sample, fit_embedding)
```

#### (3) `capvton/pipeline.py` — LeffaPipeline에 layout/fit 전달

```python
def __call__(self, ..., 
             layout_cond=None,      # 추가: (B, 2, H/8, W/8) mask+sdf
             fit_embedding=None,    # 추가: (B, 128)
):
    ...
    # latent_model_input 구성 시 layout_cond 추가
    if layout_cond is not None:
        latent_model_input = torch.cat([
            _latent_model_input,
            mask_latent, masked_image_latent, densepose_latent,
            layout_cond,  # 추가 2ch
        ], dim=1)
    
    # noise_pred = self.unet(..., fit_embedding=fit_embedding, ...)
```

#### (4) `capvton/transform.py` — LeffaTransform에 layout 데이터 처리 추가

```python
def forward(self, batch):
    ...
    # layout_mask, sdf_map 처리 추가
    if "layout_mask" in batch:
        layout_mask = process_layout(batch["layout_mask"][i])
        sdf_map = process_sdf(batch["sdf_map"][i])
        layout_cond = torch.cat([layout_mask, sdf_map], dim=0)
        layout_cond_list.append(layout_cond)
    
    batch["layout_cond"] = torch.stack(layout_cond_list)
```

#### (5) `capvton/inference.py` — LeffaInference에 fit 파라미터 전달

```python
def __call__(self, data, **kwargs):
    ...
    fit_embedding = kwargs.get("fit_embedding", None)
    layout_cond = data.get("layout_cond", None)
    
    images = self.pipe(
        ...,
        layout_cond=layout_cond,
        fit_embedding=fit_embedding,
    )
```

#### (6) `vton_script.py` — CAPVirtualTryOn 확장

```python
class CAPVirtualTryOn:
    def __init__(self, ckpt_dir):
        ...
        # 추가
        self.fit_predictor = RuleBasedFitPredictor()  # MVP
        self.layout_generator = None  # lazy-load
    
    def fit_predict(self, user_measurements, garment_measurements, size):
        """사이즈 추천 + 부위별 판정"""
        return self.fit_predictor.predict(user_measurements, garment_measurements, size)
    
    def capvton_predict(self, ..., 
                        user_measurements=None,
                        garment_measurements=None,
                        target_size=None):
        ...
        # Step 0: Fit Prediction (새로 추가)
        if user_measurements and garment_measurements:
            fit_report = self.fit_predict(user_measurements, garment_measurements, target_size)
            fit_embedding = self._encode_fit(fit_report)
            layout_mask, sdf_map = self._generate_layout(agnostic_mask, densepose, fit_embedding)
        else:
            fit_report, fit_embedding, layout_mask, sdf_map = None, None, None, None
        
        # Step 5 (기존 VT inference 수정): layout_cond 전달
        data["layout_cond"] = [layout_cond] if layout_cond else None
        result = inference(data, ..., fit_embedding=fit_embedding)
```

### 5.3 수정 위치 요약표

| 파일 | 수정 유형 | 내용 |
|------|----------|------|
| `capvton/model.py:LeffaModel.__init__` | 수정 | `new_in_channels=14`, FitFiLMAdapter 추가 |
| `capvton/model.py:LeffaModel.build_models` | 수정 | conv_in 14ch 확장 |
| `capvton/diffusion_model/unet_gen.py:forward` | 수정 | `fit_embedding`, `fit_film_adapter` 파라미터 추가, FiLM 적용 |
| `capvton/diffusion_model/unet_block_gen.py` | 수정 | CrossAttnDownBlock/UpBlock에 FiLM 전달 |
| `capvton/pipeline.py:__call__` | 수정 | `layout_cond`, `fit_embedding` 전달 |
| `capvton/transform.py:forward` | 수정 | layout 데이터 전처리 추가 |
| `capvton/inference.py:__call__` | 수정 | fit 관련 kwargs 전달 |
| `vton_script.py:CAPVirtualTryOn` | 수정 | fit_predict(), layout 생성 통합 |
| `capvton/fit/` (전체) | **신규** | Fit 모듈 전체 |

---

## 6. 평가 지표 및 실험 설계

### 6.1 핏 판정 정확도

| 지표 | 설명 | 측정 방법 |
|------|------|----------|
| **Part-wise Accuracy** | 부위별 fit class(tight/regular/loose) 정확도 | 라벨링된 테스트셋 대비 accuracy |
| **MAE(tightness)** | tightness 연속값 오차 | Mean Absolute Error |
| **Size Rec. Accuracy** | 추천 사이즈 정답률 | Top-1/Top-2 accuracy |
| **Risk Detection F1** | 리스크 부위 검출 precision/recall | F1-score |
| **Uncertainty Calibration** | 불확실성 추정 보정도 | Expected Calibration Error (ECE) |

### 6.2 실루엣 일관성

| 지표 | 설명 | 수식 |
|------|------|------|
| **Mask IoU** | 예측 마스크 vs GT 마스크 | $IoU = \frac{\|M_{pred} \cap M_{gt}\|}{\|M_{pred} \cup M_{gt}\|}$ |
| **Boundary F-score** | 경계선 정밀도 | $F = \frac{2 \cdot P_b \cdot R_b}{P_b + R_b}$ (τ=2px) |
| **SDF L1** | SDF 맵 오차 | $\frac{1}{HW}\sum\|SDF_{pred} - SDF_{gt}\|$ |
| **Length Accuracy** | 기장 픽셀 오차 | hemline y좌표 차이 (px) |

### 6.3 이미지 생성 품질

| 지표 | 설명 |
|------|------|
| **FID** | Fréchet Inception Distance (전체) |
| **KID** | Kernel Inception Distance (소규모 세트에 적합) |
| **LPIPS** | Learned Perceptual Image Patch Similarity |
| **SSIM** | 구조적 유사도 (paired GT 있을 때) |
| **CLIP-IQA** | 텍스트-이미지 일관성 (prompt 일치도) |

### 6.4 사용자 만족 / 반품 리스크 Proxy

| Proxy Metric | 설명 |
|-------------|------|
| **Fit Consistency Score** | `fit_report.tightness`와 실제 이미지에서 관측된 실루엣 차이의 상관 |
| **Wrinkle Density** | 주름 빈도 (tight 판정 영역에서 높아야 함) — edge 응답 밀도로 측정 |
| **Silhouette Gap** | body contour vs garment contour 간 거리 분포 — loose 판정 영역에서 커야 함 |
| **User Study NPS** | A/B 테스트 (기존 vs 핏 반영) 선호도 조사 |

### 6.5 Ablation 실험 계획

| 실험 | 설명 | 기대 |
|------|------|------|
| **Baseline** | 기존 CaP-VTON (핏 조건 없음) | 기준선 |
| **+Mask only** | Layout mask만 생성 → conv_in 13ch | 실루엣 범위 개선 |
| **+Mask+SDF** | Mask + SDF → conv_in 14ch | 경계 부드러움 ↑ |
| **+FiLM** | Mask+SDF + FiLM 멀티스케일 | 타이트/오버 디테일 ↑ |
| **+FiLM(no SDF)** | Mask + FiLM (SDF 없이) | SDF 기여도 검증 |
| **+Lines** | Mask+SDF+Lines → conv_in 16ch | 구조선 효과 (Phase 3) |
| **+SMPL prior** | 3D→2D projection | 상한선 확인 (Phase 3) |

---

## 7. 3단계 로드맵

### Phase 1: 추천/판정 기능 + 단순 시각화 (2~4주)

| 주차 | 태스크 | 산출물 |
|------|--------|--------|
| 1주 | 측정치 스키마 정의 + 입력 UI/API 설계 | `schema.py`, API endpoint spec |
| 1주 | Body Estimator (키포인트→추정치수) | `body_estimator.py` |
| 2주 | Rule-based Fit Predictor 구현 + 테스트 | `fit_predictor_rule.py`, 단위 테스트 |
| 2주 | 사이즈 추천 로직 + FitReport 시각화 | 텍스트/점수 출력 + 히트맵 오버레이 |
| 3주 | 기존 CaP-VTON에 fit_report 기반 마스크 스케일링 (간이 연동) | 수정된 `vton_script.py` |
| 4주 | E2E 통합 테스트 + 정성 평가 | 데모 |

**Phase 1 결과물:**
- 사이즈 추천 텍스트 (S/M/L + 부위별 판정)
- 기존 VTON에 마스크만 살짝 조절한 "간이 핏 시각화"
- Rule-based predictor 기준선 수립

### Phase 2: Layout Generator + 멀티스케일 주입 (4~6주)

| 주차 | 태스크 | 산출물 |
|------|--------|--------|
| 5주 | Pseudo-fit 데이터 증강 파이프라인 | `pseudo_augment.py`, 증강 데이터셋 |
| 5-6주 | Layout Generator 모델 설계 + 학습 | `layout_generator.py`, 체크포인트 |
| 6-7주 | FiLM Adapter 구현 + U-Net 통합 | `film_adapter.py`, 수정된 `unet_gen.py` |
| 7-8주 | conv_in 14ch 확장 + 파이프라인 연동 | 수정된 `model.py`, `pipeline.py` |
| 8-9주 | Diffusion fine-tuning (FiLM only, 기존 frozen) | fine-tuned 체크포인트 |
| 9-10주 | 정량 평가 (IoU, FID, ablation) + 정성 평가 | 실험 보고서 |

**Phase 2 결과물:**
- "핏 인지" try-on 이미지 (타이트→주름/당김, 오버→여유 공간 반영)
- Ablation 결과 (Mask only vs Mask+SDF vs +FiLM)
- 학습 기반 Fit Predictor (선택: 데이터 충분 시)

### Phase 3: 고도화 — SMPL/3D + 고급 시각화 (선택, 4~6주)

| 주차 | 태스크 | 산출물 |
|------|--------|--------|
| 11-12주 | SMPL-X 체형 추정 (사용자 치수→shape params) | SMPL 연동 모듈 |
| 12-13주 | 3D 드레이핑 시뮬레이션 (압력/air-gap 맵 생성) | 합성 신호 |
| 13-14주 | 3D 신호→2D projection → 추가 조건 채널 | 확장된 파이프라인 |
| 14-15주 | 구조선(hem/waist/sleeve lines) 추가 + 평가 | 최종 릴리즈 |
| 15-16주 | Learning-based Fit Predictor 고도화 (피드백 데이터) | 개선된 모델 |

**Phase 3 결과물:**
- SMPL 기반 정밀 체형 매칭
- 3D 근거 신호로 더 사실적인 주름/당김 표현
- 핏 판정 정확도 상한 향상

---

## 부록: 주요 설계 결정 근거

### A. 왜 FiLM인가? (vs Cross-Attention)

| 기준 | FiLM | Cross-Attention |
|------|------|-----------------|
| 파라미터 추가 | ~0.1M (linear 2개/블록) | ~2M (QKV projection/블록) |
| 기존 가중치 영향 | 없음 (plus 연산) | Self-attn 키/밸류에 토큰 추가 |
| 학습 안정성 | scale=1, shift=0 초기화로 안전 | 기존 reference feature와 충돌 가능 |
| 표현력 | 전역 조건(fit) 전달에 충분 | 공간적 조건에 더 적합 |

**결론:** fit_embedding은 전역 벡터(128-d)이므로 FiLM이 비용 대비 최적. 공간적 조건(layout)은 입력 채널 concat으로.

### B. 왜 2-stage 학습인가?

1. Layout Generator를 먼저 학습 → pseudo GT 품질 확인 가능
2. Diffusion fine-tuning 시 FiLM만 학습 → 기존 품질 유지
3. 분리 학습으로 디버깅/개선 용이

### C. 불확실성 처리 전략

- 추정 치수는 ±σ 구간으로 표현
- Fit Predictor 출력에 `confidence` 필드 포함
- UI에서: "추정 기반 결과입니다. 정확한 치수 입력 시 더 정확한 결과를 받으실 수 있습니다."
- 리스크 판정 시: 불확실 부위는 "주의" 단계로 표시 (확정 판정 회피)

