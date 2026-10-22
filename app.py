"""
Lumina AI - Backend Flask
Tương thích API chuẩn OpenAI (đã tối ưu cho Groq)
"""

import os
import json
import logging
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ============================================================
# 1. LOAD BIẾN MÔI TRƯỜNG
# ============================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LuminaAI")


# ============================================================
# 2. KHỞI TẠO FLASK APP
# ============================================================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "lumina_default_key_change_me")


# ============================================================
# 3. ĐỌC CẤU HÌNH TỪ .env
# ============================================================
API_TOKEN  = os.getenv("API_TOKEN", "").strip()
API_LINK   = os.getenv("API_LINK", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()

if not API_TOKEN or not API_LINK:
    log.warning("⚠️  Thiếu API_TOKEN hoặc API_LINK trong file .env!")
    log.warning("    → Kiểm tra lại file .env cùng thư mục với app.py")
else:
    masked = f"{API_TOKEN[:12]}...{API_TOKEN[-4:]}" if len(API_TOKEN) > 16 else "***"
    log.info(f"✅ Cấu hình OK")
    log.info(f"   Token: {masked}")
    log.info(f"   URL:   {API_LINK}")
    log.info(f"   Model: {MODEL_NAME}")


# ============================================================
# 4. LƯU LỊCH SỬ CHAT (tạm thời trong RAM)
# ============================================================
chat_history = []
MAX_HISTORY = 50


# ============================================================
# 5. ROUTE: TRANG CHỦ
# ============================================================
@app.route("/")
def index():
    """Trả về giao diện chat chính."""
    return render_template("index.html")


# ============================================================
# 6. ROUTE: XỬ LÝ CHAT
# ============================================================
@app.route("/api/chat", methods=["POST"])
def chat():
    if not API_TOKEN or not API_LINK:
        log.error("Server chưa cấu hình API_TOKEN hoặc API_LINK")
        return jsonify({
            "error": "Server chưa cấu hình API. Kiểm tra file .env."
        }), 500

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Tin nhắn không được để trống"}), 400

    if len(user_message) > 8000:
        return jsonify({"error": "Tin nhắn quá dài (tối đa 8000 ký tự)"}), 400

    log.info(f"💬 User: {user_message[:80]}{'...' if len(user_message) > 80 else ''}")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là Lumina AI, một trợ lý thông minh, thân thiện, "
                    "trả lời bằng tiếng Việt tự nhiên. "
                    "Trả lời ngắn gọn, súc tích, dùng emoji khi phù hợp."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            API_LINK,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code == 401:
            log.error("401 Unauthorized - Token sai hoặc đã bị revoke")
            return jsonify({
                "error": "API token không hợp lệ. Vào console.groq.com/keys "
                         "để tạo token mới và cập nhật .env"
            }), 401

        if response.status_code == 429:
            log.error("429 Too Many Requests - Hết quota")
            return jsonify({
                "error": "Đã hết quota miễn phí. Vui lòng đợi hoặc nâng cấp tài khoản Groq."
            }), 429

        if response.status_code == 400:
            err_body = response.text[:200]
            log.error(f"400 Bad Request: {err_body}")
            return jsonify({
                "error": f"Yêu cầu không hợp lệ. Kiểm tra MODEL_NAME. Chi tiết: {err_body}"
            }), 400

        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        log.error("Timeout khi gọi API Groq")
        return jsonify({"error": "API phản hồi quá lâu (>60s). Thử lại sau."}), 504

    except requests.exceptions.ConnectionError:
        log.error("Không kết nối được tới API Groq")
        return jsonify({
            "error": "Không kết nối được tới API. Kiểm tra mạng hoặc API_LINK."
        }), 502

    except requests.exceptions.RequestException as e:
        log.error(f"Lỗi request: {e}")
        return jsonify({"error": f"Lỗi gọi API: {str(e)}"}), 502

    except (KeyError, IndexError):
        log.error(f"Phản hồi API không đúng cấu trúc: {result}")
        return jsonify({"error": "Phản hồi API không hợp lệ."}), 502

    except json.JSONDecodeError:
        log.error("Không parse được JSON từ API")
        return jsonify({"error": "Không đọc được phản hồi API."}), 502

    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {
        "user": user_message,
        "reply": reply,
        "time": timestamp,
    }
    chat_history.append(entry)
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)

    log.info(f"⚡ Bot:  {reply[:80]}{'...' if len(reply) > 80 else ''}")

    return jsonify({"reply": reply, "timestamp": timestamp})


# ============================================================
# 7. ROUTE: LỊCH SỬ
# ============================================================
@app.route("/api/history", methods=["GET"])
def history():
    return jsonify({"history": list(reversed(chat_history[-20:]))})


# ============================================================
# 8. ROUTE: HEALTH CHECK
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "api_configured": bool(API_TOKEN and API_LINK),
        "history_count": len(chat_history),
    })


# ============================================================
# 9. CHẠY SERVER
# ============================================================
if __name__ == "__main__":
    log.info("⚡ Lumina AI đang khởi động...")
    log.info("🌐 Mở trình duyệt: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)