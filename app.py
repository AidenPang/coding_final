import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import time
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import av

# Import local modules
from src.model import MNISTClassifier
from src.hand_tracker import HandTracker
from src.data_manager import DataManager
from src.utils import preprocess_canvas_image
from train_model import train_and_save

# Set page configuration
st.set_page_config(
    page_title="AI 魔法手勢畫布與心算遊戲",
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded"
)

# Inject Premium CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1c24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #a0a5c0;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e303f !important;
        color: #00ffcc !important;
        border-bottom: 2px solid #00ffcc !important;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: #0f111a;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00cc99;
        color: #ffffff;
        box-shadow: 0px 0px 10px #00ffcc;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e303f;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'tracker' not in st.session_state:
    st.session_state.tracker = HandTracker()

# Initialize AI model weights check
weights_path = 'assets/weights.npz'
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = os.path.exists(weights_path)

if 'classifier' not in st.session_state:
    if st.session_state.model_trained:
        st.session_state.classifier = MNISTClassifier(weights_path)
    else:
        st.session_state.classifier = None

# Quiz Game States
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_streak' not in st.session_state:
    st.session_state.quiz_streak = 0
if 'quiz_question' not in st.session_state:
    st.session_state.quiz_question = ""
if 'quiz_answer' not in st.session_state:
    st.session_state.quiz_answer = 0
if 'quiz_time_start' not in st.session_state:
    st.session_state.quiz_time_start = 0.0
if 'quiz_total_time' not in st.session_state:
    st.session_state.quiz_total_time = 0.0
if 'quiz_feedback' not in st.session_state:
    st.session_state.quiz_feedback = None

# ----------------- HELPER FUNCTIONS -----------------
def generate_math_question():
    """Generates an arithmetic question whose answer is a single digit (0-9)."""
    import random
    ops = ['+', '-', '*']
    op = random.choice(ops)
    
    if op == '+':
        ans = random.randint(0, 9)
        val1 = random.randint(0, ans)
        val2 = ans - val1
        question = f"{val1} + {val2} = ?"
    elif op == '-':
        val1 = random.randint(0, 9)
        val2 = random.randint(0, val1)
        ans = val1 - val2
        question = f"{val1} - {val2} = ?"
    else: # Multiplication
        ans = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Find factors
        factors = []
        for i in range(1, 10):
            if ans % i == 0 and ans // i < 10:
                factors.append((i, ans // i))
        if not factors:
            # Fallback
            val1, val2, ans = 2, 3, 6
        else:
            val1, val2 = random.choice(factors)
        question = f"{val1} x {val2} = ?"
        
    st.session_state.quiz_question = question
    st.session_state.quiz_answer = ans
    st.session_state.quiz_time_start = time.time()
    st.session_state.quiz_feedback = None

# WebRTC Video Processor Class
# WebRTC Video Processor Class
class WebRTCHandProcessor:
    def __init__(self, question="Free Draw Mode", answer=-1, score=0, streak=0, session_id=None):
        self.tracker = HandTracker()
        self.classifier = MNISTClassifier(weights_path)
        self.canvas = None
        
        # Initialize score and stats from arguments
        self.score = score
        self.streak = streak
        
        # Initialize quiz question and answer from arguments
        self.current_question = question
        self.target_answer = answer
        self.session_id = session_id
        
        self.feedback_text = ""
        self.feedback_color = (0, 255, 0)
        self.feedback_time = 0.0
        
        # Gesture submitting status
        self.submit_frames = 0
        self.submit_cooldown = 0
        
    def generate_question_local(self):
        import random
        ops = ['+', '-', '*']
        op = random.choice(ops)
        if op == '+':
            ans = random.randint(0, 9)
            val1 = random.randint(0, ans)
            val2 = ans - val1
            self.current_question = f"{val1} + {val2} = ?"
        elif op == '-':
            val1 = random.randint(0, 9)
            val2 = random.randint(0, val1)
            ans = val1 - val2
            self.current_question = f"{val1} - {val2} = ?"
        else:
            ans = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            factors = []
            for i in range(1, 10):
                if ans % i == 0 and ans // i < 10:
                    factors.append((i, ans // i))
            if not factors:
                val1, val2, ans = 2, 3, 6
            else:
                val1, val2 = random.choice(factors)
            self.current_question = f"{val1} x {val2} = ?"
        self.target_answer = ans

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape
        
        # Initialize drawing canvas with black frame if empty
        if self.canvas is None or self.canvas.shape != (h, w):
            self.canvas = np.zeros((h, w), dtype=np.uint8)
            
        # Process hand landmarks
        processed_frame, cursor_pos, gesture = self.tracker.process_frame(img)
        
        # Draw on canvas if 'draw' gesture
        if cursor_pos is not None:
            cx, cy = cursor_pos
            if gesture == 'draw':
                # Draw thick white line on black canvas
                cv2.circle(self.canvas, (cx, cy), 14, 255, -1)
            elif gesture == 'clear':
                # Clear canvas
                self.canvas.fill(0)
                
        # Overlay the canvas onto the video frame in Cyan color
        mask = self.canvas > 0
        processed_frame[mask] = [255, 255, 0] # BGR Cyan
        
        # Hand gesture submission logic
        if self.submit_cooldown > 0:
            self.submit_cooldown -= 1
        else:
            if gesture == 'submit' and self.target_answer != -1:
                self.submit_frames += 1
                # Draw progress bar/text
                progress = min(100, int((self.submit_frames / 20.0) * 100))
                cv2.rectangle(processed_frame, (w // 2 - 100, h - 90), (w // 2 + 100, h - 70), (0, 0, 0), -1)
                cv2.rectangle(processed_frame, (w // 2 - 100, h - 90), (w // 2 - 100 + int(progress * 2), h - 70), (0, 255, 255), -1)
                cv2.putText(
                    processed_frame, 
                    f"SUBMITTING... {progress}%", 
                    (w // 2 - 80, h - 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
                
                # Triggers submission when held for ~0.6 seconds (20 frames)
                if self.submit_frames >= 20:
                    self.submit_frames = 0
                    if np.sum(self.canvas) > 0:
                        # Preprocess and Predict
                        mnist_ready = preprocess_canvas_image(self.canvas)
                        pred_digit, _ = self.classifier.predict(mnist_ready)
                        
                        if pred_digit == self.target_answer:
                            self.score += 1
                            self.streak += 1
                            self.feedback_text = f"CORRECT! Answer is {pred_digit} \xf0\x9f\x8e\x89"
                            self.feedback_color = (0, 255, 0) # Green
                            self.generate_question_local()
                        else:
                            self.streak = 0
                            self.feedback_text = f"WRONG! Recognized {pred_digit} \xe2\x9d\x8c"
                            self.feedback_color = (0, 0, 255) # Red
                            
                        self.feedback_time = time.time()
                        self.submit_cooldown = 60 # Cooldown of 2 seconds (60 frames)
                        self.canvas.fill(0) # Clear canvas automatically after submit
                        
                        # Trigger rerun to sync score and question with main thread
                        if self.session_id is not None:
                            try:
                                from streamlit.runtime import runtime
                                runtime.get_instance().request_rerun(self.session_id)
                            except Exception:
                                pass
            else:
                self.submit_frames = 0

        # Draw Banners (HUD)
        overlay = processed_frame.copy()
        # Top banner for Question
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        # Bottom banner for Score/Streak
        cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
        # Blend banners
        cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
        
        # Banners text
        cv2.putText(
            processed_frame, 
            f"QUESTION: {self.current_question}", 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 255), 
            2
        )
        
        cv2.putText(
            processed_frame, 
            f"SCORE: {self.score}  |  STREAK: {self.streak}", 
            (20, h - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        cv2.putText(
            processed_frame, 
            f"GESTURE: {gesture.upper()}", 
            (w - 220, h - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            2
        )

        # Show Large Feedback Banner in Center of Screen if active
        if time.time() - self.feedback_time < 2.0:
            overlay_fb = processed_frame.copy()
            cv2.rectangle(overlay_fb, (w // 2 - 220, h // 2 - 40), (w // 2 + 220, h // 2 + 40), self.feedback_color, -1)
            cv2.addWeighted(overlay_fb, 0.7, processed_frame, 0.3, 0, processed_frame)
            cv2.putText(
                processed_frame, 
                self.feedback_text, 
                (w // 2 - 200, h // 2 + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# ----------------- SIDEBAR STATUS -----------------
with st.sidebar:
    st.title("⚙️ 控制面板")
    st.subheader("模型訓練狀態")
    if st.session_state.model_trained:
        st.success("✅ MNIST 權重加載成功")
    else:
        st.warning("⚠️ 未偵測到模型權重！")
        if st.button("🚀 開始訓練模型 (約需 30 秒)"):
            with st.spinner("正在下載 MNIST 資料集並訓練 3-Layer MLP..."):
                try:
                    train_and_save()
                    st.session_state.model_trained = True
                    st.session_state.classifier = MNISTClassifier(weights_path)
                    st.success("🎉 訓練完畢，模型已就緒！")
                    st.rerun()
                except Exception as e:
                    st.error(f"訓練失敗: {e}")
                    
    st.write("---")
    st.write("👉 **手勢說明**：")
    st.info("""
    1. ☝️ **單指立起 (食指)**：畫布書寫模式。
    2. ✌️ **雙指立起 (食指+中指)**：確認提交答案。
    3. 🖐️ **五指張開 (手掌)**：清除整張畫布（重置）。
    4. ✊ **握拳/收回手指 (拳頭)**：無動作。
    """)
    st.write("---")
    st.caption("進階程式設計期末報告 | AI 手勢魔法畫布與心算遊戲")

# ----------------- MAIN INTERFACE -----------------
st.title("🎨 AI 魔法手勢畫布與心算挑戰")
st.markdown("結合 **MediaPipe 手勢追蹤**、**MNIST 手寫辨識** 與 **Pandas 數據分析** 的網頁應用程式")

if not st.session_state.model_trained:
    st.info("💡 **請先在左側邊欄點擊「開始訓練模型」**，訓練完成後即可開啟所有 AI 辨識功能！")
else:
    # Load WebRTC component conditional import to avoid issues during startup
    try:
        from streamlit_webrtc import webrtc_streamer
        webrtc_available = True
    except ImportError:
        webrtc_available = False
        st.error("找不到 streamlit-webrtc 庫，請確保執行 pip install -r requirements.txt")

    # Load HTML5 drawing pad library
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_pad_available = True
    except ImportError:
        canvas_pad_available = False
        st.warning("找不到 streamlit-drawable-canvas 庫，網頁手寫板功能將受限。")

    # Define Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 AI 數學心算挑戰", 
        "🎨 自由魔法畫布", 
        "📊 排行榜與數據分析", 
        "ℹ️ 專案說明與下載"
    ])

    # ==================== TAB 1: MATH QUIZ GAME ====================
    with tab1:
        st.header("🎮 AI 數學心算挑戰")
        st.write("回答畫面上出現的數學題目，並在空中或手寫板上寫出正確答案！")
        st.write("👉 **手勢答題說明**：在空中寫出答案後，比出 **雙指立起 (✌️)** 懸停 1 秒即可提交！分數將自動同步。")
        
        # Sync scores and question from the WebRTC thread to session state
        if webrtc_available and 'webrtc_ctx' in st.session_state and st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.video_processor:
            proc = st.session_state.webrtc_ctx.video_processor
            st.session_state.quiz_score = proc.score
            st.session_state.quiz_streak = proc.streak
            st.session_state.quiz_question = proc.current_question
            st.session_state.quiz_answer = proc.target_answer
            
        # Display Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("當前得分 (Score)", st.session_state.quiz_score)
        with col2:
            st.metric("連續答對 (Streak)", st.session_state.quiz_streak)
        with col3:
            st.metric("答題模式", "Webcam 手勢" if webrtc_available else "網頁手寫板")
            
        st.write("---")
        
        if not st.session_state.quiz_active:
            # Start Game Screen
            col_start, _ = st.columns([2, 2])
            with col_start:
                st.subheader("準備好開始挑戰了嗎？")
                st.write("遊戲開始後將會開始計時，請用最快的速度寫下正確答案。")
                if st.button("開始遊戲 🏁", key="start_game_btn"):
                    st.session_state.quiz_active = True
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_streak = 0
                    st.session_state.quiz_total_time = 0.0
                    generate_math_question()
                    # Sync to active video processor if exists
                    if 'webrtc_ctx' in st.session_state and st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.video_processor:
                        proc = st.session_state.webrtc_ctx.video_processor
                        proc.score = 0
                        proc.streak = 0
                        proc.current_question = st.session_state.quiz_question
                        proc.target_answer = st.session_state.quiz_answer
                        if proc.canvas is not None:
                            proc.canvas.fill(0)
                    st.rerun()
        else:
            # Active Game Screen
            st.subheader(f"❓ 請回答：:blue[{st.session_state.quiz_question}]")
            
            # Input Selection
            input_mode = st.radio("選擇輸入方式：", ["Webcam 手勢空中畫圖", "網頁手寫板 (滑鼠/觸控)"], horizontal=True)
            
            col_play_l, col_play_r = st.columns([3, 2])
            
            prediction = None
            probs = None
            canvas_img_to_predict = None
            
            with col_play_l:
                if input_mode == "Webcam 手勢空中畫圖" and webrtc_available:
                    st.write("📽️ 在鏡頭前伸出 **食指** 開始畫圖，**五指張開 (🖐️)** 清除畫布，**雙指立起 (✌️)** 提交答案。")
                    # Get session ID to sync background thread state changes with main thread
                    from streamlit.runtime.scriptrunner import get_script_run_ctx
                    ctx_info = get_script_run_ctx()
                    session_id = ctx_info.session_id if ctx_info else None

                    ctx = webrtc_streamer(
                        key="quiz-camera",
                        video_processor_factory=lambda q=st.session_state.quiz_question, a=st.session_state.quiz_answer, s=st.session_state.quiz_score, k=st.session_state.quiz_streak, sid=session_id: WebRTCHandProcessor(q, a, s, k, sid),
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"video": True, "audio": False},
                    )
                    st.session_state.webrtc_ctx = ctx
                    
                    if ctx.video_processor:
                        # Grab canvas from the WebRTC thread
                        canvas_raw = ctx.video_processor.canvas
                        if canvas_raw is not None and np.sum(canvas_raw) > 0:
                            canvas_img_to_predict = canvas_raw
                            # Display small preview of the black & white canvas
                            st.image(canvas_raw, width=150, caption="手勢畫布預覽")
                        else:
                            st.info("✍️ 提示：請在視訊畫面中書寫。")
                    else:
                        st.info("💡 請啟動上方的 Start 按鈕來開啟視訊鏡頭。")
                        
                elif input_mode == "網頁手寫板 (滑鼠/觸控)" and canvas_pad_available:
                    st.write("✏️ 用滑鼠或觸控筆在下方黑色區域寫下單個數字：")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 1)",
                        stroke_width=18,
                        stroke_color="#FFFFFF",
                        background_color="#000000",
                        width=320,
                        height=320,
                        drawing_mode="freedraw",
                        key="quiz-pad",
                    )
                    
                    if canvas_result.image_data is not None:
                        # canvas_result.image_data is RGBA
                        canvas_img_to_predict = canvas_result.image_data
            
            with col_play_r:
                st.subheader("💡 提交與判定")
                
                # Real-time Prediction Display
                if canvas_img_to_predict is not None:
                    mnist_ready_preview = preprocess_canvas_image(canvas_img_to_predict)
                    if np.sum(mnist_ready_preview) > 0:
                        pred_digit_preview, probs_preview = st.session_state.classifier.predict(mnist_ready_preview)
                        confidence_preview = probs_preview[pred_digit_preview] * 100
                        st.markdown(f"🤖 **AI 即時辨識結果**： :green[{pred_digit_preview}] (信心度: **{confidence_preview:.1f}%**)")
                        # Show 28x28 normalized MNIST feed small preview
                        st.image(mnist_ready_preview, width=100, caption="AI 眼中的圖像")
                    else:
                        st.markdown("✍️ **AI 即時辨識結果**： *請在左側畫布上書寫...*")
                else:
                    st.markdown("✍️ **AI 即時辨識結果**： *請開啟鏡頭或在手寫板上書寫...*")
                
                st.write("---")
                
                # Feedback Display
                if st.session_state.quiz_feedback:
                    st.markdown(st.session_state.quiz_feedback)
                    
                if st.button("提交答案 🚀", use_container_width=True):
                    if canvas_img_to_predict is not None:
                        # Preprocess image
                        mnist_ready = preprocess_canvas_image(canvas_img_to_predict)
                        
                        if np.sum(mnist_ready) > 0:
                            # Predict
                            prediction, probs = st.session_state.classifier.predict(mnist_ready)
                            
                            # Time taken
                            time_taken = time.time() - st.session_state.quiz_time_start
                            st.session_state.quiz_total_time += time_taken
                            
                            # Check answer
                            if prediction == st.session_state.quiz_answer:
                                st.session_state.quiz_score += 1
                                st.session_state.quiz_streak += 1
                                st.session_state.quiz_feedback = f"✅ **回答正確！** 辨識結果為 **{prediction}**。花費時間：{time_taken:.2f} 秒。"
                                generate_math_question()
                            else:
                                st.session_state.quiz_streak = 0
                                st.session_state.quiz_feedback = f"❌ **回答錯誤！** 辨識結果為 **{prediction}**，正確答案應為 **{st.session_state.quiz_answer}**。請再試一次！"
                                st.session_state.quiz_time_start = time.time() # Reset timer for retry
                                
                            # Sync back to video processor if active
                            if 'webrtc_ctx' in st.session_state and st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.video_processor:
                                proc = st.session_state.webrtc_ctx.video_processor
                                proc.score = st.session_state.quiz_score
                                proc.streak = st.session_state.quiz_streak
                                proc.current_question = st.session_state.quiz_question
                                proc.target_answer = st.session_state.quiz_answer
                                if proc.canvas is not None:
                                    proc.canvas.fill(0)
                            st.rerun()
                        else:
                            st.warning("⚠️ 請先在畫布上寫字後再提交！")
                    else:
                        st.warning("⚠️ 請先在畫布上寫字後再提交！")
                
                if st.button("跳過此題 ⏩", use_container_width=True):
                    generate_math_question()
                    # Sync to active video processor if exists
                    if 'webrtc_ctx' in st.session_state and st.session_state.webrtc_ctx and st.session_state.webrtc_ctx.video_processor:
                        proc = st.session_state.webrtc_ctx.video_processor
                        proc.current_question = st.session_state.quiz_question
                        proc.target_answer = st.session_state.quiz_answer
                        if proc.canvas is not None:
                            proc.canvas.fill(0)
                    st.rerun()
                    
                st.write("---")
                
                # End Game Form
                st.subheader("📥 結束並儲存紀錄")
                player_name = st.text_input("請輸入玩家姓名：", "Player 1")
                if st.button("結算成績並存檔 💾", use_container_width=True):
                    if st.session_state.quiz_score > 0:
                        success = st.session_state.data_manager.add_score(
                            player_name=player_name,
                            score=st.session_state.quiz_score,
                            streak=st.session_state.quiz_streak,
                            game_mode="Webcam" if input_mode == "Webcam 手勢空中畫圖" else "DrawingPad",
                            time_elapsed=round(st.session_state.quiz_total_time, 2)
                        )
                        if success:
                            st.success("🏆 成績已成功寫入排行榜！")
                            st.session_state.quiz_active = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("儲存分數失敗，請檢查資料夾權限。")
                    else:
                        st.warning("⚠️ 得分為 0 分時無法存檔，請至少答對一題！")
                        st.session_state.quiz_active = False
                        st.rerun()

    # ==================== TAB 2: FREE PAINT ====================
    with tab2:
        st.header("🎨 自由手勢魔法畫布")
        st.write("這是一個 AI 手寫辨識的沙盒環境。寫下任意數字 (0-9)，看看 AI 的辨識率！")
        
        input_mode_paint = st.radio("選擇輸入方式：", ["Webcam 空中畫布", "手寫板模式"], key="paint_mode", horizontal=True)
        
        col_paint_l, col_paint_r = st.columns([3, 2])
        
        canvas_to_recognize = None
        
        with col_paint_l:
            if input_mode_paint == "Webcam 空中畫布" and webrtc_available:
                st.write("📽️ 在鏡頭前伸出 **食指** 開始畫圖，**五指張開 (🖐️)** 清除畫布。")
                ctx_paint = webrtc_streamer(
                    key="paint-camera",
                    video_processor_factory=WebRTCHandProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                )
                
                if ctx_paint.video_processor:
                    canvas_raw = ctx_paint.video_processor.canvas
                    if canvas_raw is not None:
                        canvas_to_recognize = canvas_raw
                        st.image(canvas_raw, width=200, caption="手勢畫布即時狀態")
                else:
                    st.info("💡 請點擊 Start 按鈕啟動視訊鏡頭。")
                    
            elif input_mode_paint == "手寫板模式" and canvas_pad_available:
                st.write("✏️ 用滑鼠或觸控筆在下方黑色區域寫下單個數字：")
                canvas_result_paint = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    width=350,
                    height=350,
                    drawing_mode="freedraw",
                    key="paint-pad",
                )
                
                if canvas_result_paint.image_data is not None:
                    canvas_to_recognize = canvas_result_paint.image_data
                    
        with col_paint_r:
            st.subheader("🔮 AI 辨識結果")
            if canvas_to_recognize is not None:
                # Preprocess image
                mnist_ready = preprocess_canvas_image(canvas_to_recognize)
                
                if np.sum(mnist_ready) > 0:
                    # Predict
                    prediction, probs = st.session_state.classifier.predict(mnist_ready)
                    
                    # Display Prediction
                    st.markdown(f"### 預測數字為： :green[{prediction}]")
                    
                    # Render Confidence Bar Charts
                    prob_df = pd.DataFrame({
                        '數字': [str(i) for i in range(10)],
                        '信心度 (%)': [round(p * 100, 2) for p in probs]
                    })
                    
                    # Highlight top class
                    st.bar_chart(prob_df.set_index('數字'), height=250)
                    
                    # Show 28x28 normalized MNIST feed
                    st.subheader("👀 AI 眼中的圖像 (28x28 像素)")
                    st.image(mnist_ready, width=120, caption="經中心化與縮放處理後")
                else:
                    st.info("✍️ 請先在左側畫布上書寫，AI 將自動進行即時辨識。")
            else:
                st.info("✍️ 請先在左側畫布上書寫，AI 將自動進行即時辨識。")

    # ==================== TAB 3: STATS & LEADERBOARD ====================
    with tab3:
        st.header("📊 排行榜與數據分析")
        st.write("使用 **Pandas** 進行即時數據聚合與分析，顯示玩家歷史成績與各項指標。")
        
        # Load leaderboard
        df_leaderboard = st.session_state.data_manager.get_leaderboard(limit=50)
        analytics = st.session_state.data_manager.get_analytics()
        
        if df_leaderboard.empty:
            st.info("📭 目前排行榜尚無資料，快去玩心算挑戰賽吧！")
        else:
            # Metric Rows
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("累計遊玩次數", analytics['total_games'])
            with col_b:
                st.metric("歷史最高分數", analytics['max_score'])
            with col_c:
                st.metric("歷史最高連勝", analytics['max_streak'])
            with col_d:
                st.metric("平均答題時間", f"{analytics['avg_time_elapsed']:.2f} 秒")
                
            st.write("---")
            
            # Displays Leaderboard Table
            st.subheader("🏆 全球排行榜前 50 名")
            st.dataframe(
                df_leaderboard, 
                use_container_width=True,
                column_config={
                    "Rank": "排名",
                    "PlayerName": "玩家姓名",
                    "Score": "答對題數",
                    "Streak": "最大連勝數",
                    "GameMode": "遊玩模式",
                    "TimeElapsed": "花費總時間 (秒)",
                    "Timestamp": "遊玩時間"
                }
            )
            
            # Leaderboard Charting
            st.subheader("📈 成績趨勢分析")
            chart_col_l, chart_col_r = st.columns(2)
            
            with chart_col_l:
                # Top scores bar chart
                top_players = df_leaderboard.head(10)
                st.write("📊 **前十名得分比較**")
                st.bar_chart(top_players.set_index('PlayerName')['Score'])
                
            with chart_col_r:
                # Distribution of Game Modes
                st.write("🎮 **玩家模式選擇分佈**")
                raw_df = pd.read_csv(st.session_state.data_manager.get_raw_csv_path())
                mode_counts = raw_df['GameMode'].value_counts()
                st.bar_chart(mode_counts)

    # ==================== TAB 4: ABOUT & DOWNLOADS ====================
    with tab4:
        st.header("ℹ️ 專案說明與數據下載")
        
        col_desc_l, col_desc_r = st.columns(2)
        
        with col_desc_l:
            st.subheader("💡 專案架構")
            st.markdown("""
            此專案展示了高級 Python 程式設計的多項關鍵知識點：
            1. **物件導向設計 (OOP)**：`MNISTClassifier` 用於神經網路運算、`HandTracker` 包裝 MediaPipe 手勢偵測、`DataManager` 包裝 Pandas 資料讀寫。
            2. **NumPy 矩陣運算**：
               - 在 `src/model.py` 中，使用純 NumPy 實作 MLP Forward 運算（矩陣點積、ReLU、Softmax）。
               - 在 `src/utils.py` 中，利用 OpenCV/NumPy 的 bounding box 計算，將使用者手繪的邊界框裁切、等比例縮放為 20x20，並置中放在 28x28 圖像內。
            3. **Pandas 資料分析**：在 `data_manager.py` 讀寫 CSV 檔，使用 Pandas 計算平均時間、最大連勝數，並自動排序排行榜。
            4. **例外處理 (Exception Handling)**：
               - 自動捕捉模型權重缺失錯誤，引導使用者點擊邊欄訓練模型。
               - 自動捕捉攝影機加載失敗，允許無攝影機時降級成網頁手寫板模式。
            """)
            
        with col_desc_r:
            st.subheader("📥 數據導出與對接")
            st.write("為了實現網頁版與單機版 (Pygame) 的數據聯通，您可以將目前的排行榜數據導出為 CSV 檔。")
            
            raw_csv_path = st.session_state.data_manager.get_raw_csv_path()
            try:
                with open(raw_csv_path, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
                    
                st.download_button(
                    label="📥 下載排行榜數據 (leaderboard.csv)",
                    data=csv_data,
                    file_name="leaderboard.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("點擊上方按鈕下載後，將檔案放入未來單機版專案的 `data/` 資料夾下，即可與網頁版共享排行數據！")
            except Exception as e:
                st.error(f"加載資料庫失敗: {e}")
                
            st.write("---")
            st.subheader("🧑‍🏫 專案程式碼目錄結構")
            st.code("""
ai_magic_canvas_game/
├── train_model.py          # MNIST 模型訓練 (PyTorch)
├── app.py                  # Streamlit 網頁主要介面
├── requirements.txt        # 專案套件依賴
├── data/
│   └── leaderboard.csv     # 儲存玩家成績
├── assets/
│   └── weights.npz         # 導出的權重矩陣
└── src/
    ├── model.py            # NumPy AI 推論類別
    ├── hand_tracker.py     # MediaPipe 手勢偵測
    ├── data_manager.py     # Pandas 排行榜數據庫管理
    └── utils.py            # NumPy 圖像置中預處理
            """)
