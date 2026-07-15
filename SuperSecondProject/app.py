import os
import io
import json
import base64
from pathlib import Path

import faiss
import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms


# ===================================================
# 0. Hugging Face Space 기준 파일 경로
# ===================================================
# Space Files 탭의 루트 구조가 아래와 같아야 합니다.
#
# app.py
# requirements.txt
# models/
#   officehome_convnext_phase_center_best.pt
# officehome_caption_project/
#   data/
#     officehome_search_metadata.csv
#   faiss_index/
#     officehome_caption_config.json
#     officehome_caption.index

PROJECT_DIR = Path("officehome_caption_project")
DATA_DIR = PROJECT_DIR / "data"
FAISS_DIR = PROJECT_DIR / "faiss_index"
MODEL_DIR = Path("models")

CONFIG_PATH = FAISS_DIR / "officehome_caption_config.json"
CNN_CHECKPOINT_PATH = MODEL_DIR / "officehome_convnext_phase_center_best.pt"


# ===================================================
# 1. 경로 확인 유틸
# ===================================================
def must_exist(path: Path, label: str) -> Path:
    """파일이 없을 때 Hugging Face 로그에서 바로 원인을 볼 수 있게 에러를 명확히 냅니다."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} 파일을 찾을 수 없습니다: {path}\n"
            f"현재 작업 위치: {Path.cwd()}\n"
            "Hugging Face Files 탭에서 파일 구조를 확인하세요."
        )
    return path


def resolve_existing_path(config_value, fallback_paths):
    """
    config json 안 경로가 Colab 절대경로(/content/drive/...)로 남아 있어도,
    Hugging Face Space 내부 상대경로 후보에서 실제 존재하는 파일을 찾아 사용합니다.
    """
    candidates = []

    if config_value:
        config_path = Path(config_value)
        candidates.append(config_path)

    candidates.extend([Path(p) for p in fallback_paths])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "필요 파일을 찾지 못했습니다.\n"
        "확인한 후보 경로:\n"
        + "\n".join([f"- {str(c)}" for c in candidates])
        + f"\n현재 작업 위치: {Path.cwd()}"
    )


# ===================================================
# 2. Caption Search 로드
# ===================================================
CAPTION_SEARCH_LOADED = False
caption_model = None
caption_df = None
caption_index = None


def load_caption_search_once():
    global CAPTION_SEARCH_LOADED
    global caption_model, caption_df, caption_index

    if CAPTION_SEARCH_LOADED:
        print("Caption search는 이미 로드되어 있습니다.")
        return caption_model, caption_df, caption_index

    must_exist(CONFIG_PATH, "Caption config")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    embedding_model_name = config["embedding_model_name"]

    metadata_path = resolve_existing_path(
        config.get("metadata_path"),
        [
            DATA_DIR / "officehome_search_metadata.csv",
            PROJECT_DIR / "officehome_search_metadata.csv",
            FAISS_DIR / "officehome_search_metadata.csv",
        ],
    )

    faiss_index_path = resolve_existing_path(
        config.get("faiss_index_path"),
        [
            FAISS_DIR / "officehome_caption.index",
        ],
    )

    print("Caption config:", CONFIG_PATH)
    print("Metadata path:", metadata_path)
    print("FAISS index path:", faiss_index_path)

    caption_model = SentenceTransformer(embedding_model_name)
    caption_df = pd.read_csv(metadata_path)
    caption_index = faiss.read_index(str(faiss_index_path))

    CAPTION_SEARCH_LOADED = True

    print("Caption search 로드 완료")
    print("Embedding model:", embedding_model_name)
    print("Metadata shape:", caption_df.shape)
    print("FAISS vector count:", caption_index.ntotal)

    return caption_model, caption_df, caption_index


# Lazy loading: Space 시작 시 바로 로드하지 않고, 첫 검색 시 로드합니다.
# caption_model, caption_df, caption_index = load_caption_search_once()


# ===================================================
# 3. CNN 모델 로드
# ===================================================
# Lazy loading: Space 시작 시 바로 CNN을 로드하지 않고,
# 이미지 검색 버튼을 처음 눌렀을 때 1회만 로드합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_LOADED = False
cnn_model = None
cnn_val_transform = None
classes = None


def load_cnn_once():
    global CNN_LOADED
    global cnn_model, cnn_val_transform, classes

    if CNN_LOADED:
        print("CNN 모델은 이미 로드되어 있습니다.")
        return cnn_model, cnn_val_transform, classes

    must_exist(CNN_CHECKPOINT_PATH, "CNN checkpoint")

    checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    cnn_model = models.convnext_tiny(weights=None)

    in_features = cnn_model.classifier[2].in_features

    # 학습 때 저장된 key가 classifier.2.1.weight 형태였으므로
    # 단순 Linear가 아니라 Sequential(Dropout, Linear) 구조로 맞춥니다.
    cnn_model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    cnn_val_transform = transforms.Compose(
        [
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    CNN_LOADED = True

    print("CNN 모델 로드 완료")
    print("클래스 수:", len(classes))

    return cnn_model, cnn_val_transform, classes


# ===================================================
# 4. Caption 검색 본체 함수
# ===================================================
def search_caption(query, top_k=5):
    # 첫 텍스트 검색 시 Caption 모델/metadata/FAISS index를 1회 로드합니다.
    load_caption_search_once()

    if query is None or str(query).strip() == "":
        return pd.DataFrame({"message": ["검색어를 입력해주세요."]})

    top_k = int(top_k)

    query_vec = caption_model.encode(
        [str(query)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = caption_index.search(query_vec, top_k)

    results = caption_df.iloc[indices[0]].copy()
    results["score"] = scores[0]

    cols = [
        "score",
        "object",
        "domain",
        "size",
        "price_usd",
        "caption",
        "image_url",
        "source_filepath",
    ]

    available_cols = [c for c in cols if c in results.columns]
    results = results[available_cols]

    if "score" in results.columns:
        results["score"] = results["score"].round(4)

    return results


# 기존 이름을 쓰는 코드가 있어도 동작하도록 alias 유지
def search_caption_gradio(query, top_k=5):
    return search_caption(query, top_k)


# ===================================================
# 5. CNN 이미지 분류 함수
# ===================================================
def predict_image_class(pil_image):
    # 첫 이미지 검색 시 CNN 모델을 1회 로드합니다.
    load_cnn_once()

    if pil_image is None:
        return None, None

    image = pil_image.convert("RGB")
    image_tensor = cnn_val_transform(image).unsqueeze(0).to(device)

    cnn_model.eval()
    with torch.no_grad():
        logits = cnn_model(image_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = classes[pred_idx.item()]
    confidence = float(conf.item())

    return pred_class, confidence


# ===================================================
# 6. HTML 출력 유틸
# ===================================================
def pil_to_base64(pil_image, max_size=(220, 220)):
    img = pil_image.copy()
    img.thumbnail(max_size)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def safe_value(row, column, default=""):
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def dataframe_to_result_html(results_df, title="검색 결과", extra_info=""):
    if results_df is None or len(results_df) == 0:
        return f"""
        <div style="padding:16px; border:1px solid #ddd; border-radius:12px;">
            <h3>{title}</h3>
            <p>검색 결과가 없습니다.</p>
            <div>{extra_info}</div>
        </div>
        """

    html = f"""
    <div style="padding:16px;">
        <h3>{title}</h3>
        <div style="margin-bottom:12px;">{extra_info}</div>
    """

    for _, row in results_df.iterrows():
        image_url = safe_value(row, "image_url")
        score = safe_value(row, "score")
        obj = safe_value(row, "object")
        domain = safe_value(row, "domain")
        size = safe_value(row, "size")
        price_usd = safe_value(row, "price_usd")
        caption = safe_value(row, "caption")
        source_filepath = safe_value(row, "source_filepath")

        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = str(score)

        if image_url:
            image_html = f"""
            <img src="{image_url}" style="width:180px; border-radius:10px; border:1px solid #eee;" />
            """
        else:
            image_html = """
            <div style="width:180px; height:120px; border:1px solid #eee; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#888;">
                No image
            </div>
            """

        html += f"""
    <div style="
        border:1px solid #ddd;
        border-radius:14px;
        padding:14px;
        margin-bottom:14px;
        display:flex;
        gap:16px;
        align-items:flex-start;
        background:#fafafa;
        color:#222222;
    ">
        <div style="min-width:180px;">
            {image_html}
        </div>
        <div style="
            flex:1;
            color:#222222;
            font-size:14px;
            line-height:1.5;
        ">
            <p style="color:#222222;"><b style="color:#111111;">score</b>: {score_text}</p>
            <p style="color:#222222;"><b style="color:#111111;">object</b>: {obj}</p>
            <p style="color:#222222;"><b style="color:#111111;">domain</b>: {domain}</p>
            <p style="color:#222222;"><b style="color:#111111;">size</b>: {size}</p>
            <p style="color:#222222;"><b style="color:#111111;">price_usd</b>: {price_usd}</p>
            <p style="color:#222222;"><b style="color:#111111;">caption</b>: {caption}</p>
            <p style="color:#555555;"><b style="color:#333333;">source</b>: {source_filepath}</p>
        </div>
    </div>
"""

    html += "</div>"
    return html


# ===================================================
# 7. Gradio 연결 함수
# ===================================================
def search_caption_text_gradio(query, top_k):
    if query is None or str(query).strip() == "":
        return """
        <div style="padding:16px; border:1px solid #ddd; border-radius:12px;">
            검색어를 입력해주세요.
        </div>
        """

    top_k = int(top_k)
    results = search_caption(query, top_k=top_k)

    extra_info = f"""
    <div style="padding:10px; background:#f3f4f6; border-radius:10px;">
        <b>입력 검색어:</b> {query}
    </div>
    """

    return dataframe_to_result_html(results, title="텍스트 검색 결과", extra_info=extra_info)


def search_image_and_caption_gradio(image, query, top_k):
    if image is None and (query is None or str(query).strip() == ""):
        return """
        <div style="padding:16px; border:1px solid #ddd; border-radius:12px;">
            이미지 또는 검색어 중 하나 이상 입력해주세요.
        </div>
        """

    pred_class, confidence = None, None
    image_html = ""

    if image is not None:
        pred_class, confidence = predict_image_class(image)
        img_b64 = pil_to_base64(image)

        image_html = f"""
        <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;">
            <img src="{img_b64}" style="width:180px; border-radius:12px; border:1px solid #ddd;" />
            <div style="padding:10px; background:#eef6ff; border-radius:10px;">
                <p><b>CNN 예측 클래스:</b> {pred_class}</p>
                <p><b>예측 확률:</b> {confidence:.4f}</p>
            </div>
        </div>
        """

    query = "" if query is None else str(query).strip()

    if pred_class is not None and query != "":
        final_query = f"{query}, {pred_class}"
    elif pred_class is not None:
        final_query = pred_class
    else:
        final_query = query

    top_k = int(top_k)
    results = search_caption(final_query, top_k=top_k)

    extra_info = f"""
    {image_html}
    <div style="padding:10px; background:#f3f4f6; border-radius:10px;">
        <p><b>사용자 입력 문장:</b> {query if query != '' else '(없음)'}</p>
        <p><b>최종 검색어:</b> {final_query}</p>
    </div>
    """

    return dataframe_to_result_html(results, title="이미지 + 텍스트 검색 결과", extra_info=extra_info)


# ===================================================
# 8. Gradio UI
# ===================================================
with gr.Blocks(title="OfficeHome CNN + Caption Search") as demo:
    gr.Markdown(
        """
        # OfficeHome CNN + Caption Search

        OfficeHome 이미지 분류 CNN과 SentenceTransformer + FAISS 캡션 검색을 결합한 데모입니다.

        - **텍스트 검색**: 자연어 검색어로 캡션/메타데이터 검색
        - **이미지 + 텍스트 검색**: 업로드 이미지를 CNN으로 먼저 분류한 뒤, 예측 클래스를 검색어에 결합

        모델은 앱 시작 시점이 아니라 첫 검색 버튼 클릭 시 로드됩니다.
        첫 실행은 다소 시간이 걸릴 수 있습니다.
        """
    )

    with gr.Tabs():
        with gr.Tab("이미지 + 텍스트 검색"):
            with gr.Row():
                image_input = gr.Image(
                    type="pil",
                    label="이미지 업로드",
                )

                with gr.Column():
                    image_query_input = gr.Textbox(
                        label="추가 검색어",
                        placeholder="예: black monitor, office desk item, transparent bottle",
                        lines=2,
                    )

                    image_top_k_input = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=5,
                        step=1,
                        label="검색 결과 개수",
                    )

                    image_search_button = gr.Button("이미지 기반 검색하기")

            image_result_html = gr.HTML(label="이미지 + 텍스트 검색 결과")

            image_search_button.click(
                fn=search_image_and_caption_gradio,
                inputs=[image_input, image_query_input, image_top_k_input],
                outputs=image_result_html,
            )

        with gr.Tab("텍스트 검색"):
            text_query_input = gr.Textbox(
                label="검색어",
                placeholder="예: monitor, laptop, desk lamp, transparent bottle",
                lines=2,
            )

            text_top_k_input = gr.Slider(
                minimum=1,
                maximum=12,
                value=5,
                step=1,
                label="검색 결과 개수",
            )

            text_search_button = gr.Button("텍스트 검색하기")

            text_result_html = gr.HTML(label="텍스트 검색 결과")

            text_search_button.click(
                fn=search_caption_text_gradio,
                inputs=[text_query_input, text_top_k_input],
                outputs=text_result_html,
            )


# Hugging Face Space에서는 share=True를 사용하지 않습니다.
demo.launch()
