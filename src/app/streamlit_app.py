import base64
import sys
import random
from io import BytesIO
from pathlib import Path
import requests
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from retrieval.retrieval_logic import FurnitureRetriever, CATEGORY_CHAINS

DEFAULT_ROOM = "bedrooms"
UPLOAD_ID    = "__uploaded__"

ROOM_LABELS = {
    "bedrooms":    "🛏️  Bedrooms",
    "living_rooms": "🛋️  Living Rooms",
}

STEP_LABELS = {
    "bed":           "🛏️  Bed",
    "sofa":          "🛋️  Sofa",
    "small_storage": "📦  Small Storage",
    "large_storage": "🗄️  Large Storage",
    "table":         "🪑  Table",
    "chair_stool":   "💺  Chair / Stool",
    "curtain":       "🪟  Curtain",
}

@st.cache_resource(show_spinner="Loading retrieval model...")
def load_retriever(room: str) -> FurnitureRetriever:
    return FurnitureRetriever(room=room, embed_weight=0.8, hist_weight=0.2)


def reset():
    """Clear furniture selection but keep room choice."""
    st.session_state.step     = 0
    st.session_state.selected = []   # list of chosen item dicts
    st.session_state.skipped  = []   # list of category names that were skipped
    st.session_state.options  = []
    st.session_state.exclude  = set()
    st.session_state.pop("uploaded_item",  None)
    st.session_state.pop("uploaded_emb",   None)
    st.session_state.pop("uploaded_hist",  None)


def full_reset():
    """Go back to room selection screen."""
    st.session_state.pop("room", None)
    st.session_state.pop("room_chosen", None)
    reset()


def init():
    if "step" not in st.session_state:
        reset()


def current_chain() -> list[str]:
    return CATEGORY_CHAINS[st.session_state.get("room", DEFAULT_ROOM)]


def _uploaded_cat() -> str | None:
    item = st.session_state.get("uploaded_item")
    return item["category"] if item else None


def advance(retriever: FurnitureRetriever):
    """Move to the next step and pre-load options for it."""
    chain = current_chain()
    st.session_state.step += 1
    st.session_state.options = []

    # Auto-skip the uploaded category (already filled by the user's photo)
    up_cat = _uploaded_cat()
    if up_cat and st.session_state.step < len(chain) and chain[st.session_state.step] == up_cat:
        st.session_state.step += 1

    if st.session_state.step < len(chain):
        next_cat = chain[st.session_state.step]
        st.session_state.options = retriever.get_compatible(
            selected        = st.session_state.selected,
            target_category = next_cat,
            top_k           = 5,
            exclude_ids     = st.session_state.exclude,
        )


def pick(item: dict, retriever: FurnitureRetriever):
    st.session_state.selected.append(item)
    st.session_state.exclude.add(item["furniture_id"])
    advance(retriever)
    st.rerun()


def skip(cat: str, retriever: FurnitureRetriever):
    st.session_state.skipped.append(cat)
    advance(retriever)
    st.rerun()


def scene_image_path(item: dict, room: str) -> Path:
    return (BASE_DIR / "data" / "processed_data" / room
            / item["source"] / item["scene"] / "scene_image.jpg")


def show_item_meta(item: dict, room: str = DEFAULT_ROOM):
    st.caption(f"ID: {item.get('furniture_id', '-')}")
    href = item.get("furniture_href")
    if href:
        st.link_button("Open product page", href, width='stretch')
    google_lens_button(item["image_path"], key=item["furniture_id"])
    scene_path = scene_image_path(item, room)
    if scene_path.exists():
        with st.expander("📷 Original scene"):
            st.image(str(scene_path), width='stretch')


def show_progress():
    chain   = current_chain()
    step    = st.session_state.step
    skipped = st.session_state.get("skipped", [])
    up_cat  = _uploaded_cat()

    cols = st.columns(len(chain))
    for i, cat in enumerate(chain):
        label = STEP_LABELS[cat]
        if i < step:
            if cat == up_cat:
                cols[i].info(f"📷 {label}")
            elif cat in skipped:
                cols[i].warning(f"~~{label}~~")
            else:
                cols[i].success(label)
        elif i == step:
            cols[i].info(f"**{label}**")
        else:
            cols[i].markdown(f"<div style='color:#aaa'>{label}</div>",
                             unsafe_allow_html=True)


def img_b64(path: str) -> str:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def img_bytes(path: str) -> bytes:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def google_lens_button(image_path: str, key: str) -> None:
    """Generates a public URL for the local image and passes it to Google Lens."""
    state_key = f"lens_url_{key}"
    
    # If we already generated the link for this specific item, show the 'Open' button
    if state_key in st.session_state:
        st.link_button(
            "🌐 Open in Google Lens", 
            st.session_state[state_key], 
            use_container_width=True
        )
    else:
        # Show the 'Generate' button
        if st.button("🔍 Find similar (Google Lens)", key=f"btn_{key}", use_container_width=True):
            with st.spinner("Preparing Lens link..."):
                try:
                    # Upload to tmpfiles.org (allows Googlebot to fetch the image)
                    with open(image_path, "rb") as f:
                        res = requests.post(
                            "https://tmpfiles.org/api/v1/upload", 
                            files={"file": f}
                        )
                    res.raise_for_status()
                    
                    # Convert the viewer URL to a direct raw image download URL
                    data = res.json()
                    viewer_url = data["data"]["url"]
                    public_url = viewer_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    
                    # Construct the official Google Lens URL query
                    lens_url = f"https://lens.google.com/uploadbyurl?url={urllib.parse.quote(public_url)}"
                    
                    # Save to session state and reload to show the actual link button
                    st.session_state[state_key] = lens_url
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate link. Error: {e}")


def make_collage(items: list[dict], cell_size: int = 300, padding: int = 12,
                 label_height: int = 28) -> bytes:
    """Horizontal strip collage of all selected items with category labels."""
    from PIL import ImageDraw, ImageFont
    n = len(items)
    w = n * cell_size + (n + 1) * padding
    h = cell_size + label_height + padding * 2
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, item in enumerate(items):
        img = Image.open(item["image_path"]).convert("RGB")
        img.thumbnail((cell_size, cell_size), Image.LANCZOS)
        x = padding + i * (cell_size + padding)
        y = padding + (cell_size - img.height) // 2
        canvas.paste(img, (x, y))
        label = STEP_LABELS.get(item["category"], item["category"])
        if item["furniture_id"] == UPLOAD_ID:
            label = f"📷 {label}"
        draw.text((x, padding + cell_size + 4), label, fill=(80, 80, 80), font=font)

    buf = BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def dominant_colors(path: str, n: int = 5) -> list[tuple]:
    """Return n dominant RGB colors sorted by frequency."""
    img = Image.open(path).convert("RGB").resize((60, 60))
    paletted = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    counts: dict[int, int] = {}
    for idx in paletted.getdata():
        counts[idx] = counts.get(idx, 0) + 1
    palette = paletted.getpalette()
    return [
        (palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2])
        for i in sorted(counts, key=lambda x: -counts[x])[:n]
    ]


def show_color_swatches(path: str, n: int = 5) -> None:
    colors = dominant_colors(path, n)
    swatches = "".join(
        f'<span style="display:inline-block;width:22px;height:22px;'
        f'background:rgb({r},{g},{b});border-radius:4px;margin:1px;'
        f'border:1px solid #ccc"></span>'
        for r, g, b in colors
    )
    st.sidebar.markdown(f'<div style="margin:2px 0 6px 0">{swatches}</div>',
                        unsafe_allow_html=True)


def show_image(path: str, width: str = "100%") -> None:
    st.markdown(
        f'<img src="data:image/jpeg;base64,{img_b64(path)}" '
        f'style="width:{width};object-fit:contain;">',
        unsafe_allow_html=True,
    )


def show_options(items: list[dict], retriever: FurnitureRetriever, room: str = DEFAULT_ROOM):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            show_image(item["image_path"])
            if item["score"] != 0.0:
                st.caption(
                    f"score: **{item['score']:.3f}**  "
                    f"(E:{item['embed_score']:.2f} / B:{item['hist_score']:.2f})"
                )
            show_item_meta(item, room)
            cat_slug = item["category"].replace("_", "-")
            st.download_button(
                "⬇️ Download",
                data=img_bytes(item["image_path"]),
                file_name=f"{cat_slug}_{item['furniture_id']}.jpg",
                mime="image/jpeg",
                key=f"dl_opt_{item['furniture_id']}",
                width='stretch',
            )
            if st.button("Select", key=f"sel_{item['furniture_id']}"):
                pick(item, retriever)


def show_selected_sidebar():
    st.sidebar.header("Your room so far")
    skipped = st.session_state.get("skipped", [])

    if not st.session_state.selected and not skipped:
        st.sidebar.caption("Nothing selected yet.")
        return

    for item in st.session_state.selected:
        st.sidebar.markdown(
            f'<img src="data:image/jpeg;base64,{img_b64(item["image_path"])}" '
            f'style="width:120px;object-fit:contain;">',
            unsafe_allow_html=True,
        )
        if item["furniture_id"] == UPLOAD_ID:
            st.sidebar.caption(f"📷 Your photo — {STEP_LABELS[item['category']]}")
        else:
            st.sidebar.caption(STEP_LABELS[item["category"]])
            st.sidebar.caption(f"ID: {item.get('furniture_id', '-')}")
            st.sidebar.caption(f"Scene: {item.get('scene', '-')}")
            if item.get("furniture_href"):
                st.sidebar.link_button("Open product page", item["furniture_href"],
                                       width='stretch')
        show_color_swatches(item["image_path"])
        st.sidebar.divider()

    if skipped:
        st.sidebar.caption(f"Skipped: {', '.join(STEP_LABELS[c] for c in skipped)}")


def show_final_room():
    room   = st.session_state.get("room", DEFAULT_ROOM)
    selected = st.session_state.selected
    skipped  = st.session_state.get("skipped", [])

    st.header("🏠 Your Room")

    if selected:
        cols = st.columns(len(selected))
        for col, item in zip(cols, selected):
            with col:
                show_image(item["image_path"])
                if item["furniture_id"] == UPLOAD_ID:
                    st.caption(f"📷 {STEP_LABELS[item['category']]} (your photo)")
                else:
                    st.caption(STEP_LABELS[item["category"]])
                    show_item_meta(item, room)
                cat_slug = item["category"].replace("_", "-")
                st.download_button(
                    "⬇️ Download",
                    data=img_bytes(item["image_path"]),
                    file_name=f"{cat_slug}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{item['furniture_id']}",
                    width='stretch',
                )

        st.divider()
        collage = make_collage(selected)
        st.download_button(
            "⬇️ Download collage (all items)",
            data=collage,
            file_name="room_collage.jpg",
            mime="image/jpeg",
            width='stretch'
        )
    else:
        st.info("You skipped all categories — nothing to show!")

    if skipped:
        st.caption(f"Skipped categories: {', '.join(STEP_LABELS[c] for c in skipped)}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart (same room)"):
            reset()
            st.rerun()
    with col2:
        if st.button("🏠 Choose different room"):
            full_reset()
            st.rerun()


def show_upload_section(retriever: FurnitureRetriever, chain: list[str]):
    """Optional expander at step 0 — lets user seed the chain with their own photo."""
    with st.expander("📷 Start with your own photo (optional)"):
        st.caption(
            "Upload a furniture photo and the chain will build recommendations around it. "
            "The chain still starts from step 1 — your photo is the anchor."
        )
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"], key="photo_upload"
        )
        cat_labels  = [STEP_LABELS[c] for c in chain]
        chosen_label = st.selectbox("This item is a:", cat_labels, key="upload_cat_select")
        chosen_cat   = chain[cat_labels.index(chosen_label)]

        if uploaded_file is not None:
            pil = Image.open(uploaded_file).convert("RGB")
            st.image(pil, width=180)

            if st.button("✅ Use this photo as anchor", key="use_photo_btn"):
                tmp_dir = BASE_DIR / "data" / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / "uploaded_anchor.jpg"
                pil.save(str(tmp_path), "JPEG", quality=90)

                with st.spinner("Embedding your photo..."):
                    emb, hist = retriever.embed_image(pil)

                retriever.register_external(UPLOAD_ID, emb, hist)

                st.session_state.uploaded_item = {
                    "furniture_id": UPLOAD_ID,
                    "category":     chosen_cat,
                    "image_path":   str(tmp_path),
                    "source":       "upload",
                    "scene":        "upload",
                    "image_name":   "uploaded_anchor.jpg",
                    "furniture_href": None,
                    "score":        0.0,
                    "embed_score":  0.0,
                    "hist_score":   0.0,
                }
                st.session_state.uploaded_emb  = emb
                st.session_state.uploaded_hist = hist
                st.session_state.selected      = [st.session_state.uploaded_item]
                st.session_state.exclude.add(UPLOAD_ID)
                st.session_state.options = []
                st.rerun()


def show_room_selection():
    st.title("🛋️ Room Constructor")
    st.markdown("### Choose a room type to get started")
    st.divider()

    cols = st.columns(2)
    for col, (room_key, room_label) in zip(cols, ROOM_LABELS.items()):
        with col:
            if st.button(room_label, width='stretch', key=f"room_{room_key}"):
                st.session_state.room = room_key
                st.session_state.room_chosen = True
                reset()
                st.rerun()


def main():
    st.set_page_config(page_title="Room Constructor", layout="wide",
                       initial_sidebar_state="expanded")

    # Room not yet chosen — show selection screen
    if not st.session_state.get("room_chosen"):
        show_room_selection()
        return

    init()

    room      = st.session_state.get("room", DEFAULT_ROOM)
    retriever = load_retriever(room)
    chain     = current_chain()

    # Keep external embedding registered (or cleared) on every rerun
    if st.session_state.get("uploaded_emb") is not None:
        retriever.register_external(
            UPLOAD_ID,
            st.session_state.uploaded_emb,
            st.session_state.uploaded_hist,
        )
    else:
        retriever.clear_external(UPLOAD_ID)

    # If the uploaded category lands on the current step, skip it immediately
    up_cat = _uploaded_cat()
    if up_cat and st.session_state.step < len(chain) and chain[st.session_state.step] == up_cat:
        st.session_state.step += 1
        st.session_state.options = []

    # Sidebar
    with st.sidebar:
        st.caption(f"Room: **{ROOM_LABELS[room]}**")
        if st.button("← Change room"):
            full_reset()
            st.rerun()
        st.divider()
        st.markdown("**Scoring mode**")
        embed_w = st.slider(
            "Style ◀─────────────▶ Color",
            0.0, 1.0, 0.8, 0.05,
            key="embed_w",
            help="Left = style/shape similarity (embeddings). Right = color similarity (histogram).",
        )
        hist_w = round(1.0 - embed_w, 2)
        st.caption(f"Style: **{embed_w:.2f}**  |  Color: **{hist_w:.2f}**")

        retriever.embed_weight = embed_w
        retriever.hist_weight  = hist_w

        # Re-fetch options when weight changes and items are already selected
        if (st.session_state.get("_last_embed_w", embed_w) != embed_w
                and st.session_state.get("selected")):
            st.session_state.options = []
        st.session_state["_last_embed_w"] = embed_w

        st.divider()
        show_selected_sidebar()

    st.title("🛋️ Room Constructor")
    st.caption(f"{ROOM_LABELS[room]} — pick items step by step. "
               "Recommendations are based on what you already chose.")

    show_progress()
    st.divider()

    step = st.session_state.step

    if step >= len(chain):
        show_final_room()
        return

    current_cat = chain[step]
    st.subheader(f"Step {step + 1} of {len(chain)} — {STEP_LABELS[current_cat]}")

    # Show upload section at step 0 only if no photo has been uploaded yet
    if step == 0 and not st.session_state.get("uploaded_item"):
        show_upload_section(retriever, chain)
        st.divider()

    # Load initial options if not yet set
    if not st.session_state.options:
        if st.session_state.selected:
            st.session_state.options = retriever.get_compatible(
                selected        = st.session_state.selected,
                target_category = current_cat,
                top_k           = 5,
                exclude_ids     = st.session_state.exclude,
            )
        else:
            st.session_state.options = retriever.get_random(current_cat, n=5)
            st.caption("ℹ️ First item is random — scoring weights apply from the next step.")

    show_options(st.session_state.options, retriever, room)

    st.divider()
    col_shuffle, col_skip = st.columns([3, 1])

    with col_shuffle:
        if st.button("🔀 Show different options"):
            if not st.session_state.selected:
                st.session_state.options = retriever.get_random(current_cat, n=5)
            else:
                pool = retriever.get_compatible(
                    selected        = st.session_state.selected,
                    target_category = current_cat,
                    top_k           = 10,
                    exclude_ids     = st.session_state.exclude,
                )
                st.session_state.options = random.sample(pool, min(5, len(pool)))
            st.rerun()

    with col_skip:
        if st.button(f"⏭️ Skip {STEP_LABELS[current_cat].split()[1]}", width='stretch'):
            skip(current_cat, retriever)


if __name__ == "__main__":
    main()