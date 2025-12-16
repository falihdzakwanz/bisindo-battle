"""
BISINDO BATTLE - Interactive Pygame Application
Game Edukasi Bahasa Isyarat Indonesia

Controls:
- Arrow Keys: Navigate menu
- Enter: Select/Confirm
- ESC: Back/Pause
- 🔥 AUTO-SUBMIT: Hold gesture konsisten 2 detik = otomatis benar!
- Space: Manual submit (optional, masih bisa dipakai)
- D: Toggle debug mode (show landmarks)
- F11: Toggle fullscreen

Game Modes:
1. Challenge Mode: Random letter, timed challenges
2. Practice Mode: Free practice with instant feedback
3. Time Attack: Score as many as possible in 60 seconds
"""

import pygame
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import random
import time
from pathlib import Path
from collections import deque

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# ============================================
# CONSTANTS & CONFIGURATION
# ============================================

# Screen settings (will be updated if fullscreen)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
FULLSCREEN = False

# Colors (Material Design inspired)
COLOR_BG = (18, 18, 18)  # Dark background
COLOR_PRIMARY = (33, 150, 243)  # Blue
COLOR_SUCCESS = (76, 175, 80)  # Green
COLOR_ERROR = (244, 67, 54)  # Red
COLOR_WARNING = (255, 193, 7)  # Yellow
COLOR_TEXT = (255, 255, 255)  # White
COLOR_TEXT_DIM = (158, 158, 158)  # Gray
COLOR_CARD = (33, 33, 33)  # Card background

# Game settings
CONFIDENCE_THRESHOLD = 0.80
PRACTICE_TIME_LIMIT = 180  # 3 minutes
TIME_ATTACK_DURATION = 60  # 60 seconds
CHALLENGE_ROUNDS = 10

# Model paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_final.onnx"
MODEL_DATA_PATH = BASE_DIR / "models" / "multimodal_final.onnx.data"

# Check if model exists
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Please run training first or download the model.")

# Class names
CLASS_NAMES = [chr(65 + i) for i in range(26)]  # A-Z

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================
# PYGAME SETUP
# ============================================

# Get display info for fullscreen
display_info = pygame.display.Info()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BISINDO BATTLE - Game Edukasi Bahasa Isyarat")
clock = pygame.time.Clock()


def toggle_fullscreen():
    """Toggle between fullscreen and windowed mode"""
    global screen, SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN
    FULLSCREEN = not FULLSCREEN
    if FULLSCREEN:
        SCREEN_WIDTH = display_info.current_w
        SCREEN_HEIGHT = display_info.current_h
        screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN
        )
    else:
        SCREEN_WIDTH = 1280
        SCREEN_HEIGHT = 720
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    print(
        f"Fullscreen: {'ON' if FULLSCREEN else 'OFF'} - {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
    )


# Fonts
font_title = pygame.font.Font(None, 72)
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)


# ============================================
# MODEL & MEDIAPIPE INITIALIZATION
# ============================================

print("🚀 Loading BISINDO Multi-Modal Model...")
try:
    ort_session = ort.InferenceSession(str(MODEL_PATH))
    image_input_name = ort_session.get_inputs()[0].name
    landmarks_input_name = ort_session.get_inputs()[1].name
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Initialize MediaPipe Hands
print("🤚 Initializing MediaPipe Hands (2-hand support)...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands for BISINDO signs
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("✅ MediaPipe initialized (2-hand mode)!")

# Initialize webcam
print("📹 Initializing webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
    exit(1)
print("✅ Webcam ready!")


# ============================================
# HELPER FUNCTIONS
# ============================================


def preprocess_image(image):
    """Preprocess image untuk inference"""
    # Resize
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-1
    image = image.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    image = (image - MEAN) / STD

    # Transpose (H, W, C) → (C, H, W)
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def detect_and_extract_landmarks(image_rgb):
    """Detect hand dan extract landmarks (supports up to 2 hands)"""
    results = hands_detector.process(image_rgb)

    if results.multi_hand_landmarks:
        # 🔥 EXTRACT LANDMARKS UNTUK HINGGA 2 TANGAN (126 features)
        all_landmarks = []
        num_hands = len(results.multi_hand_landmarks)

        # Extract up to 2 hands
        for hand_idx in range(min(2, num_hands)):
            hand_landmarks = results.multi_hand_landmarks[hand_idx]
            wrist = hand_landmarks.landmark[0]

            # Normalize relative to wrist
            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Padding if only 1 hand detected (63 features → 126 features)
        if num_hands == 1:
            all_landmarks.extend([0.0] * 63)

        landmarks = np.array(
            all_landmarks[:126], dtype=np.float32
        )  # Ensure 126 features

        # Get bounding box for first hand (for cropping)
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image_rgb.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add margin
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)

        x_min = max(0, x_min - margin_x)
        x_max = min(w, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h, y_max + margin_y)

        # Crop hand region
        cropped = image_rgb[y_min:y_max, x_min:x_max]

        # Calculate bounding boxes for ALL hands
        all_bboxes = []
        for hand_lm in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_lm.landmark]
            y_coords = [lm.y * h for lm in hand_lm.landmark]

            x_min_h = int(min(x_coords))
            x_max_h = int(max(x_coords))
            y_min_h = int(min(y_coords))
            y_max_h = int(max(y_coords))

            # Add margin
            margin_x_h = int((x_max_h - x_min_h) * 0.2)
            margin_y_h = int((y_max_h - y_min_h) * 0.2)

            x_min_h = max(0, x_min_h - margin_x_h)
            x_max_h = min(w, x_max_h + margin_x_h)
            y_min_h = max(0, y_min_h - margin_y_h)
            y_max_h = min(h, y_max_h + margin_y_h)

            all_bboxes.append((x_min_h, y_min_h, x_max_h, y_max_h))

        # Return all hands for visualization
        return (
            cropped,
            landmarks,
            results.multi_hand_landmarks,
            all_bboxes,
            True,
            num_hands,
        )

    return None, None, None, None, False, 0


def predict_gesture(image_tensor, landmarks_tensor):
    """Run inference dengan multimodal model"""
    try:
        outputs = ort_session.run(
            None,
            {image_input_name: image_tensor, landmarks_input_name: landmarks_tensor},
        )
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Get top prediction
        top_idx = np.argmax(probabilities)
        top_letter = CLASS_NAMES[top_idx]
        top_confidence = probabilities[top_idx]

        return top_letter, top_confidence, probabilities

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, 0.0, None


def draw_text(surface, text, pos, font, color=COLOR_TEXT, center=False):
    """Draw text pada surface"""
    text_surface = font.render(text, True, color)
    if center:
        text_rect = text_surface.get_rect(center=pos)
        surface.blit(text_surface, text_rect)
    else:
        surface.blit(text_surface, pos)


def draw_button(surface, text, rect, font, is_selected=False):
    """Draw button dengan highlight jika selected"""
    color = COLOR_PRIMARY if is_selected else COLOR_CARD
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, COLOR_PRIMARY, rect, 2, border_radius=10)

    draw_text(surface, text, rect.center, font, COLOR_TEXT, center=True)


def draw_card(surface, rect, color=COLOR_CARD, border_color=None, border_width=0):
    """Draw card dengan rounded corners"""
    pygame.draw.rect(surface, color, rect, border_radius=15)
    if border_color and border_width > 0:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=15)


def draw_progress_bar(surface, rect, progress, color=COLOR_PRIMARY):
    """Draw progress bar"""
    # Background
    pygame.draw.rect(surface, COLOR_CARD, rect, border_radius=5)

    # Progress
    if progress > 0:
        progress_rect = rect.copy()
        progress_rect.width = int(rect.width * progress)
        pygame.draw.rect(surface, color, progress_rect, border_radius=5)

    # Border
    pygame.draw.rect(surface, COLOR_TEXT_DIM, rect, 2, border_radius=5)


# ============================================
# GAME STATE MANAGEMENT
# ============================================


class GameState:
    MENU = "menu"
    MODE_SELECT = "mode_select"
    CHALLENGE = "challenge"
    PRACTICE = "practice"
    TIME_ATTACK = "time_attack"
    RESULTS = "results"
    PAUSED = "paused"


class Game:
    def __init__(self):
        self.state = GameState.MENU
        self.previous_state = None

        # Menu navigation
        self.menu_index = 0
        self.menu_options = ["MULAI", "CARA BERMAIN", "KELUAR"]
        self.mode_index = 0
        self.mode_options = ["Challenge Mode", "Practice Mode", "Time Attack"]

        # Game data
        self.current_letter = None
        self.score = 0
        self.round = 0
        self.total_rounds = CHALLENGE_ROUNDS
        self.start_time = 0
        self.time_limit = 0
        self.correct_answers = 0
        self.attempts = 0

        # History untuk smooth feedback
        self.prediction_history = deque(maxlen=5)

        # 🔥 AUTO-SUBMIT: Track prediksi konsisten
        self.consistent_prediction = None  # Huruf yang konsisten
        self.consistent_start_time = None  # Waktu mulai konsisten
        self.auto_submit_duration = 2.0  # 2 detik konsisten = auto submit
        self.last_auto_submit_time = 0  # Cooldown untuk avoid double submit

        # Debug mode
        self.debug_mode = False

        # Results data
        self.results = {"mode": "", "score": 0, "total": 0, "accuracy": 0.0, "time": 0}

    def reset_game(self, mode):
        """Reset game state untuk mode baru"""
        self.score = 0
        self.round = 0
        self.correct_answers = 0
        self.attempts = 0
        self.start_time = time.time()
        self.prediction_history.clear()

        # Reset auto-submit tracking
        self.consistent_prediction = None
        self.consistent_start_time = None
        self.last_auto_submit_time = 0

        if mode == GameState.CHALLENGE:
            self.total_rounds = CHALLENGE_ROUNDS
            self.time_limit = 0  # No time limit
        elif mode == GameState.PRACTICE:
            self.time_limit = PRACTICE_TIME_LIMIT
        elif mode == GameState.TIME_ATTACK:
            self.time_limit = TIME_ATTACK_DURATION

        self.generate_new_letter()

    def generate_new_letter(self):
        """Generate random letter untuk challenge"""
        self.current_letter = random.choice(CLASS_NAMES)
        self.round += 1
        # Reset auto-submit tracking saat dapat letter baru
        self.consistent_prediction = None
        self.consistent_start_time = None

    def check_auto_submit(self, prediction, confidence, current_time):
        """🔥 Check jika prediksi konsisten selama 2 detik → auto submit"""
        # Cek cooldown (avoid double submit)
        if current_time - self.last_auto_submit_time < 0.5:
            return False

        # Prediksi harus sama dengan target dan confidence tinggi
        if prediction == self.current_letter and confidence >= CONFIDENCE_THRESHOLD:
            # Jika ini prediksi konsisten pertama atau berbeda dari sebelumnya
            if self.consistent_prediction != prediction:
                self.consistent_prediction = prediction
                self.consistent_start_time = current_time
                return False

            # Cek berapa lama sudah konsisten
            duration = current_time - self.consistent_start_time
            if duration >= self.auto_submit_duration:
                # AUTO SUBMIT!
                self.last_auto_submit_time = current_time
                return True
        else:
            # Prediksi berubah atau confidence rendah → reset
            self.consistent_prediction = None
            self.consistent_start_time = None

        return False


# Initialize game
game = Game()

print("\n" + "=" * 60)
print("🎮 BISINDO BATTLE READY!")
print("=" * 60)
print("\n📖 Controls:")
print("   Arrow Keys: Navigate")
print("   Enter: Select")
print("   🔥 AUTO-SUBMIT: Hold gesture 2 detik = otomatis benar!")
print("   Space: Manual submit (optional)")
print("   D: Toggle debug mode (show landmarks)")
print("   F11: Toggle FULLSCREEN ⬅️ TEKAN INI!")
print("   ESC: Back/Pause")
print("\n✨ Selamat bermain dan belajar BISINDO!")
print("=" * 60)
print("\n🟢 BOUNDING BOX akan muncul otomatis saat tangan terdeteksi")
print("🔴 LANDMARKS (titik + garis) muncul saat tekan D (debug mode)")
print("🔥 PROGRESS BAR hijau = tangan konsisten, tunggu 2 detik!")
print("=" * 60)
print("\n⚠️  CATATAN PENTING:")
print("   ✋✋ MediaPipe BISA deteksi 2 tangan sekaligus!")
print("   🤖 TAPI model AI hanya dilatih dengan 1 tangan")
print("   📊 Prediksi menggunakan tangan PERTAMA yang terdeteksi")
print("   🔮 Untuk huruf 2 tangan: Perlu training model baru")
print("=" * 60)


# ============================================
# GAME LOOP
# ============================================

running = True
last_frame = None
last_prediction = None
last_confidence = 0.0
hand_detected = False
landmarks_vis = None
all_bboxes = []  # List to store all bounding boxes

while running:
    dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

    # Read webcam frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Mirror
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame_rgb.copy()

        # Detect hand and landmarks (now supports 2 hands)
        (
            cropped,
            landmarks,
            all_hand_landmarks,
            bbox_result,
            hand_detected,
            num_hands,
        ) = detect_and_extract_landmarks(frame_rgb)

        # Update visualization data when hand is detected
        if hand_detected:
            all_bboxes = bbox_result  # Now a list of bboxes for all hands
            landmarks_vis = all_hand_landmarks  # This is now a list of all hands!
            # Show number of hands detected
            if num_hands > 1:
                print(f"✋✋ {num_hands} tangan terdeteksi!")
            # Only print occasionally (not every frame)
        else:
            # Clear bbox when no hand
            all_bboxes = []
            landmarks_vis = None

        if hand_detected and cropped is not None and cropped.size > 0:
            # Preprocess
            image_tensor = preprocess_image(cropped)
            landmarks_tensor = landmarks.reshape(1, 126).astype(
                np.float32
            )  # 🔥 126 features (2 hands)

            # Predict
            letter, confidence, probs = predict_gesture(image_tensor, landmarks_tensor)

            if letter:
                last_prediction = letter
                last_confidence = confidence
                game.prediction_history.append((letter, confidence))
                # Print prediction (not every frame, only when confident enough)
                if confidence > 0.5:
                    print(f"🎯 {letter} ({confidence*100:.1f}%)")

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            # Debug toggle
            if event.key == pygame.K_d:
                game.debug_mode = not game.debug_mode
                print(f"\n{'='*60}")
                print(f"🔍 Debug mode: {'ON' if game.debug_mode else 'OFF'}")
                print(f"   hand_detected: {hand_detected}")
                print(f"   landmarks_vis: {'Ada' if landmarks_vis else 'None'}")
                print(f"   all_bboxes: {all_bboxes if all_bboxes else 'None'}")
                print(f"   Game state: {game.state}")
                print(f"{'='*60}\n")

            # Fullscreen toggle
            elif event.key == pygame.K_F11:
                toggle_fullscreen()

            # ESC - Back/Pause
            elif event.key == pygame.K_ESCAPE:
                if game.state == GameState.MENU:
                    running = False
                elif game.state in [
                    GameState.CHALLENGE,
                    GameState.PRACTICE,
                    GameState.TIME_ATTACK,
                ]:
                    game.previous_state = game.state
                    game.state = GameState.PAUSED
                elif game.state == GameState.PAUSED:
                    game.state = game.previous_state
                else:
                    game.state = GameState.MENU

            # Navigation
            elif event.key == pygame.K_UP:
                if game.state == GameState.MENU:
                    game.menu_index = (game.menu_index - 1) % len(game.menu_options)
                elif game.state == GameState.MODE_SELECT:
                    game.mode_index = (game.mode_index - 1) % len(game.mode_options)

            elif event.key == pygame.K_DOWN:
                if game.state == GameState.MENU:
                    game.menu_index = (game.menu_index + 1) % len(game.menu_options)
                elif game.state == GameState.MODE_SELECT:
                    game.mode_index = (game.mode_index + 1) % len(game.mode_options)

            # Selection
            elif event.key == pygame.K_RETURN:
                if game.state == GameState.MENU:
                    if game.menu_index == 0:  # MULAI
                        game.state = GameState.MODE_SELECT
                    elif game.menu_index == 1:  # CARA BERMAIN
                        pass  # TODO: Show tutorial
                    elif game.menu_index == 2:  # KELUAR
                        running = False

                elif game.state == GameState.MODE_SELECT:
                    if game.mode_index == 0:
                        game.state = GameState.CHALLENGE
                        game.reset_game(GameState.CHALLENGE)
                    elif game.mode_index == 1:
                        game.state = GameState.PRACTICE
                        game.reset_game(GameState.PRACTICE)
                    elif game.mode_index == 2:
                        game.state = GameState.TIME_ATTACK
                        game.reset_game(GameState.TIME_ATTACK)

                elif game.state == GameState.RESULTS:
                    game.state = GameState.MENU

                elif game.state == GameState.PAUSED:
                    game.state = game.previous_state

            # Space - Capture/Submit
            elif event.key == pygame.K_SPACE:
                if game.state in [
                    GameState.CHALLENGE,
                    GameState.PRACTICE,
                    GameState.TIME_ATTACK,
                ]:
                    if hand_detected and last_prediction:
                        game.attempts += 1

                        # Check if correct
                        if (
                            last_prediction == game.current_letter
                            and last_confidence >= CONFIDENCE_THRESHOLD
                        ):
                            game.correct_answers += 1
                            game.score += int(last_confidence * 100)

                            # Generate new letter
                            if game.state == GameState.CHALLENGE:
                                if game.round >= game.total_rounds:
                                    # End game
                                    game.results = {
                                        "mode": "Challenge Mode",
                                        "score": game.score,
                                        "total": game.total_rounds,
                                        "accuracy": (
                                            game.correct_answers / game.total_rounds
                                        )
                                        * 100,
                                        "time": time.time() - game.start_time,
                                    }
                                    game.state = GameState.RESULTS
                                else:
                                    game.generate_new_letter()
                            else:
                                game.generate_new_letter()

    # Update game logic
    if game.state in [GameState.CHALLENGE, GameState.PRACTICE, GameState.TIME_ATTACK]:
        elapsed = time.time() - game.start_time

        # Check time limit
        if game.time_limit > 0 and elapsed >= game.time_limit:
            # Time's up
            game.results = {
                "mode": game.state,
                "score": game.score,
                "total": game.attempts,
                "accuracy": (game.correct_answers / max(game.attempts, 1)) * 100,
                "time": elapsed,
            }
            game.state = GameState.RESULTS

        # 🔥 AUTO-SUBMIT: Check jika prediksi konsisten selama 2 detik
        if (
            hand_detected
            and last_prediction
            and last_confidence >= CONFIDENCE_THRESHOLD
        ):
            current_time = time.time()
            if game.check_auto_submit(last_prediction, last_confidence, current_time):
                # Auto submit berhasil!
                game.attempts += 1
                game.correct_answers += 1
                game.score += int(last_confidence * 100)

                print(f"✅ AUTO-SUBMIT: {last_prediction} ({last_confidence*100:.1f}%)")

                # Generate new letter
                if game.state == GameState.CHALLENGE:
                    if game.round >= game.total_rounds:
                        # End game
                        game.results = {
                            "mode": "Challenge Mode",
                            "score": game.score,
                            "total": game.total_rounds,
                            "accuracy": (game.correct_answers / game.total_rounds)
                            * 100,
                            "time": time.time() - game.start_time,
                        }
                        game.state = GameState.RESULTS
                    else:
                        game.generate_new_letter()
                else:
                    game.generate_new_letter()

    # ============================================
    # RENDERING
    # ============================================

    screen.fill(COLOR_BG)

    # Render based on game state
    if game.state == GameState.MENU:
        # Title
        draw_text(
            screen,
            "BISINDO BATTLE",
            (SCREEN_WIDTH // 2, 120),
            font_title,
            COLOR_PRIMARY,
            center=True,
        )
        draw_text(
            screen,
            "Game Edukasi Bahasa Isyarat Indonesia",
            (SCREEN_WIDTH // 2, 180),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )

        # Menu buttons
        for i, option in enumerate(game.menu_options):
            y = 300 + i * 100
            rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, y - 35, 400, 70)
            draw_button(screen, option, rect, font_large, i == game.menu_index)

        # Footer
        draw_text(
            screen,
            "Gunakan ↑↓ untuk navigasi, ENTER untuk pilih",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 70),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )
        draw_text(
            screen,
            f"Debug Mode: {'ON' if game.debug_mode else 'OFF'} (tekan D)",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 45),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )
        draw_text(
            screen,
            f"Fullscreen: F11 | Resolusi: {SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 20),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )

    elif game.state == GameState.MODE_SELECT:
        # Title
        draw_text(
            screen,
            "PILIH MODE PERMAINAN",
            (SCREEN_WIDTH // 2, 100),
            font_title,
            COLOR_PRIMARY,
            center=True,
        )

        # Mode cards
        mode_descriptions = [
            ("Challenge Mode", "Tunjukkan 10 huruf berurutan", "🎯 10 rounds"),
            ("Practice Mode", "Latihan bebas 3 menit", "🎓 Learn pace"),
            ("Time Attack", "60 detik kompetitif!", "⚡ Fast game"),
        ]

        card_width = 350
        card_height = 200
        spacing = 30
        start_x = (SCREEN_WIDTH - (card_width * 3 + spacing * 2)) // 2

        for i, (mode, desc, detail) in enumerate(mode_descriptions):
            x = start_x + i * (card_width + spacing)
            y = 250

            rect = pygame.Rect(x, y, card_width, card_height)

            # Card
            color = COLOR_PRIMARY if i == game.mode_index else COLOR_CARD
            border = 3 if i == game.mode_index else 1
            draw_card(
                screen,
                rect,
                color if i == game.mode_index else COLOR_CARD,
                COLOR_PRIMARY,
                border,
            )

            # Text
            title_color = COLOR_TEXT if i == game.mode_index else COLOR_TEXT
            draw_text(
                screen,
                mode,
                (x + card_width // 2, y + 50),
                font_medium,
                title_color,
                center=True,
            )
            draw_text(
                screen,
                desc,
                (x + card_width // 2, y + 100),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )
            draw_text(
                screen,
                detail,
                (x + card_width // 2, y + 140),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )

        # Instructions
        draw_text(
            screen,
            "←→ atau ↑↓ untuk memilih | ENTER untuk mulai | ESC kembali",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )

    elif game.state == GameState.CHALLENGE:
        # Webcam feed (left)
        webcam_size = (480, 360)
        webcam_pos = (50, 120)

        if last_frame is not None:
            frame_resized = cv2.resize(last_frame, webcam_size)

            h_orig, w_orig = last_frame.shape[:2]
            scale_x = webcam_size[0] / w_orig
            scale_y = webcam_size[1] / h_orig

            # ALWAYS draw bounding boxes for ALL hands detected
            if all_bboxes:
                colors_bbox = [
                    (0, 255, 0),
                    (255, 165, 0),
                ]  # Green for hand 1, Orange for hand 2
                for idx, bbox in enumerate(all_bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    color = colors_bbox[idx % 2]

                    # Draw thick bounding box
                    cv2.rectangle(
                        frame_resized,
                        (int(x_min * scale_x), int(y_min * scale_y)),
                        (int(x_max * scale_x), int(y_max * scale_y)),
                        color,
                        3,
                    )

                    # Add "HAND X DETECTED" text
                    text = f"HAND {idx + 1}" if len(all_bboxes) > 1 else "HAND DETECTED"
                    cv2.putText(
                        frame_resized,
                        text,
                        (int(x_min * scale_x), int(y_min * scale_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Draw landmarks ONLY in debug mode (now supports multiple hands!)
            if game.debug_mode and landmarks_vis and all_bboxes:
                h_orig, w_orig = last_frame.shape[:2]
                scale_x = webcam_size[0] / w_orig
                scale_y = webcam_size[1] / h_orig

                mp_hands_module = mp.solutions.hands
                colors = [
                    (0, 255, 255),
                    (255, 0, 255),
                ]  # Cyan for hand 1, Magenta for hand 2

                # Draw all detected hands
                for hand_idx, hand_landmarks in enumerate(landmarks_vis):
                    color = colors[hand_idx % 2]

                    # Draw connections first
                    for connection in mp_hands_module.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        start_lm = hand_landmarks.landmark[start_idx]
                        end_lm = hand_landmarks.landmark[end_idx]
                        start_x = int(start_lm.x * w_orig * scale_x)
                        start_y = int(start_lm.y * h_orig * scale_y)
                        end_x = int(end_lm.x * w_orig * scale_x)
                        end_y = int(end_lm.y * h_orig * scale_y)
                        cv2.line(
                            frame_resized,
                            (start_x, start_y),
                            (end_x, end_y),
                            color,  # Different color per hand
                            3,
                        )

                    # Draw all 21 landmarks on top
                    for lm in hand_landmarks.landmark:
                        cx = int(lm.x * w_orig * scale_x)
                        cy = int(lm.y * h_orig * scale_y)
                        cv2.circle(
                            frame_resized, (cx, cy), 8, (255, 0, 0), -1
                        )  # Red filled
                        cv2.circle(
                            frame_resized, (cx, cy), 8, (255, 255, 255), 2
                        )  # White border
            elif game.debug_mode:
                if not landmarks_vis:
                    print("⚠️  Debug ON tapi landmarks_vis = None")
                if not all_bboxes:
                    print("⚠️  Debug ON tapi all_bboxes = None")

            frame_surf = pygame.surfarray.make_surface(np.rot90(frame_resized))
            screen.blit(frame_surf, webcam_pos)
            pygame.draw.rect(
                screen,
                COLOR_TEXT_DIM,
                pygame.Rect(
                    webcam_pos[0], webcam_pos[1], webcam_size[0], webcam_size[1]
                ),
                2,
                border_radius=10,
            )

        # Game info (right)
        right_x = 600

        draw_text(screen, "CHALLENGE MODE", (right_x, 50), font_large, COLOR_PRIMARY)
        draw_text(
            screen,
            f"Round {game.round} / {game.total_rounds}",
            (right_x, 100),
            font_medium,
            COLOR_TEXT,
        )

        # Score card
        score_rect = pygame.Rect(right_x, 150, 300, 80)
        draw_card(screen, score_rect, COLOR_CARD, COLOR_PRIMARY, 2)
        draw_text(
            screen,
            "SCORE",
            (right_x + 150, 165),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )
        draw_text(
            screen,
            str(game.score),
            (right_x + 150, 200),
            font_large,
            COLOR_SUCCESS,
            center=True,
        )

        # Target letter
        target_rect = pygame.Rect(right_x, 260, 300, 200)
        draw_card(screen, target_rect, COLOR_PRIMARY)
        draw_text(
            screen,
            "TUNJUKKAN:",
            (right_x + 150, 280),
            font_small,
            COLOR_TEXT,
            center=True,
        )
        draw_text(
            screen,
            game.current_letter,
            (right_x + 150, 360),
            font_title,
            COLOR_TEXT,
            center=True,
        )

        # Prediction
        pred_rect = pygame.Rect(right_x, 490, 300, 120)

        if hand_detected and last_prediction:
            is_correct = last_prediction == game.current_letter
            pred_color = (
                COLOR_SUCCESS
                if is_correct and last_confidence >= CONFIDENCE_THRESHOLD
                else COLOR_WARNING
            )

            draw_card(screen, pred_rect, COLOR_CARD, pred_color, 3)
            draw_text(
                screen,
                "PREDIKSI:",
                (right_x + 150, 505),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )
            draw_text(
                screen,
                last_prediction,
                (right_x + 150, 545),
                font_large,
                pred_color,
                center=True,
            )

            conf_rect = pygame.Rect(right_x + 30, 590, 240, 15)
            draw_progress_bar(screen, conf_rect, last_confidence, pred_color)
            draw_text(
                screen,
                f"{last_confidence*100:.1f}%",
                (right_x + 150, 605),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )

            # 🔥 AUTO-SUBMIT PROGRESS: Tampilkan progress bar jika konsisten
            if (
                game.consistent_prediction == game.current_letter
                and game.consistent_start_time is not None
            ):
                current_time = time.time()
                elapsed_consistent = current_time - game.consistent_start_time
                progress = min(elapsed_consistent / game.auto_submit_duration, 1.0)

                # Progress bar untuk auto-submit
                auto_rect = pygame.Rect(right_x, 625, 300, 8)
                pygame.draw.rect(screen, COLOR_CARD, auto_rect, border_radius=4)

                # Fill berdasarkan progress
                fill_width = int(300 * progress)
                if fill_width > 0:
                    fill_rect = pygame.Rect(right_x, 625, fill_width, 8)
                    pygame.draw.rect(screen, COLOR_SUCCESS, fill_rect, border_radius=4)

                # Text hint
                draw_text(
                    screen,
                    f"🔥 Hold steady... {progress*100:.0f}%",
                    (right_x + 150, 645),
                    font_small,
                    COLOR_SUCCESS,
                    center=True,
                )
        else:
            draw_card(screen, pred_rect, COLOR_CARD, COLOR_TEXT_DIM, 2)
            draw_text(
                screen,
                "Tidak ada tangan",
                (right_x + 150, 545),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )

        # Instructions
        inst_y = 680
        draw_text(
            screen,
            "🔥 Hold 2 detik = Auto Submit | ESC: Pause",
            (right_x + 150, inst_y),
            font_small,
            COLOR_TEXT,
            center=True,
        )

    elif game.state == GameState.PRACTICE or game.state == GameState.TIME_ATTACK:
        elapsed = time.time() - game.start_time
        remaining = max(0, game.time_limit - elapsed)

        # Timer bar
        timer_rect = pygame.Rect(50, 20, SCREEN_WIDTH - 100, 40)
        if game.state == GameState.TIME_ATTACK:
            progress = elapsed / game.time_limit
            draw_progress_bar(
                screen,
                timer_rect,
                progress,
                COLOR_ERROR if remaining < 10 else COLOR_PRIMARY,
            )

        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        timer_text = (
            f"{minutes:02d}:{seconds:02d}"
            if game.state == GameState.PRACTICE
            else f"{int(remaining)}s"
        )
        timer_color = COLOR_ERROR if remaining < 10 else COLOR_TEXT
        draw_text(
            screen,
            timer_text,
            (SCREEN_WIDTH // 2, 40),
            font_large,
            timer_color,
            center=True,
        )

        # Webcam
        webcam_size = (640, 480)
        webcam_pos = ((SCREEN_WIDTH - webcam_size[0]) // 2, 90)

        if last_frame is not None:
            frame_resized = cv2.resize(last_frame, webcam_size)

            h_orig, w_orig = last_frame.shape[:2]
            scale_x = webcam_size[0] / w_orig
            scale_y = webcam_size[1] / h_orig

            # ALWAYS draw bounding boxes for ALL hands detected
            if all_bboxes:
                colors_bbox = [(0, 255, 0), (255, 165, 0)]  # Green & Orange
                for idx, bbox in enumerate(all_bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    color = colors_bbox[idx % 2]

                    # Draw thick bounding box
                    cv2.rectangle(
                        frame_resized,
                        (int(x_min * scale_x), int(y_min * scale_y)),
                        (int(x_max * scale_x), int(y_max * scale_y)),
                        color,
                        3,
                    )

                    # Add "HAND X DETECTED" text
                    text = f"HAND {idx + 1}" if len(all_bboxes) > 1 else "HAND DETECTED"
                    cv2.putText(
                        frame_resized,
                        text,
                        (int(x_min * scale_x), int(y_min * scale_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            # Draw landmarks ONLY in debug mode (supports 2 hands!)
            if game.debug_mode and landmarks_vis and all_bboxes:
                h_orig, w_orig = last_frame.shape[:2]
                scale_x = webcam_size[0] / w_orig
                scale_y = webcam_size[1] / h_orig

                mp_hands_module = mp.solutions.hands
                colors = [(0, 255, 255), (255, 0, 255)]  # Cyan & Magenta

                # Draw all detected hands
                for hand_idx, hand_landmarks in enumerate(landmarks_vis):
                    color = colors[hand_idx % 2]

                    # Draw connections
                    for connection in mp_hands_module.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        start_lm = hand_landmarks.landmark[start_idx]
                        end_lm = hand_landmarks.landmark[end_idx]
                        start_x = int(start_lm.x * w_orig * scale_x)
                        start_y = int(start_lm.y * h_orig * scale_y)
                        end_x = int(end_lm.x * w_orig * scale_x)
                        end_y = int(end_lm.y * h_orig * scale_y)
                        cv2.line(
                            frame_resized,
                            (start_x, start_y),
                            (end_x, end_y),
                            color,
                            3,
                        )

                    # Draw landmarks
                    for lm in hand_landmarks.landmark:
                        cx = int(lm.x * w_orig * scale_x)
                        cy = int(lm.y * h_orig * scale_y)
                        cv2.circle(
                            frame_resized, (cx, cy), 8, (255, 0, 0), -1
                        )  # Red filled
                        cv2.circle(
                            frame_resized, (cx, cy), 8, (255, 255, 255), 2
                        )  # White border

            frame_surf = pygame.surfarray.make_surface(np.rot90(frame_resized))
            screen.blit(frame_surf, webcam_pos)
            pygame.draw.rect(
                screen,
                COLOR_TEXT_DIM,
                pygame.Rect(
                    webcam_pos[0], webcam_pos[1], webcam_size[0], webcam_size[1]
                ),
                2,
                border_radius=10,
            )

        # Stats
        info_y = webcam_pos[1] + webcam_size[1] + 20
        stats_width = 200
        stats = [
            ("Benar", game.correct_answers, COLOR_SUCCESS),
            ("Total", game.attempts, COLOR_TEXT),
            ("Score", game.score, COLOR_PRIMARY),
        ]

        total_width = len(stats) * stats_width
        start_x = (SCREEN_WIDTH - total_width) // 2

        for i, (label, value, color) in enumerate(stats):
            x = start_x + i * stats_width
            rect = pygame.Rect(x, info_y, stats_width - 20, 70)
            draw_card(screen, rect, COLOR_CARD)
            draw_text(
                screen,
                label,
                (x + (stats_width - 20) // 2, info_y + 15),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )
            draw_text(
                screen,
                str(value),
                (x + (stats_width - 20) // 2, info_y + 45),
                font_medium,
                color,
                center=True,
            )

        # Target
        challenge_y = info_y + 90
        draw_text(
            screen,
            "TUNJUKKAN:",
            (SCREEN_WIDTH // 2, challenge_y),
            font_medium,
            COLOR_TEXT,
            center=True,
        )
        draw_text(
            screen,
            game.current_letter,
            (SCREEN_WIDTH // 2, challenge_y + 45),
            font_title,
            COLOR_PRIMARY,
            center=True,
        )

        # Prediction
        if hand_detected and last_prediction:
            is_match = (
                last_prediction == game.current_letter
                and last_confidence >= CONFIDENCE_THRESHOLD
            )
            pred_color = COLOR_SUCCESS if is_match else COLOR_TEXT_DIM
            pred_text = f"{last_prediction} ({last_confidence*100:.0f}%)"
            draw_text(
                screen,
                pred_text,
                (SCREEN_WIDTH // 2, challenge_y + 110),
                font_small,
                pred_color,
                center=True,
            )

            # 🔥 AUTO-SUBMIT PROGRESS
            if (
                game.consistent_prediction == game.current_letter
                and game.consistent_start_time is not None
            ):
                current_time = time.time()
                elapsed_consistent = current_time - game.consistent_start_time
                progress = min(elapsed_consistent / game.auto_submit_duration, 1.0)

                # Progress bar
                bar_width = 400
                auto_rect = pygame.Rect(
                    (SCREEN_WIDTH - bar_width) // 2, challenge_y + 140, bar_width, 10
                )
                pygame.draw.rect(screen, COLOR_CARD, auto_rect, border_radius=5)

                fill_width = int(bar_width * progress)
                if fill_width > 0:
                    fill_rect = pygame.Rect(
                        (SCREEN_WIDTH - bar_width) // 2,
                        challenge_y + 140,
                        fill_width,
                        10,
                    )
                    pygame.draw.rect(screen, COLOR_SUCCESS, fill_rect, border_radius=5)

                draw_text(
                    screen,
                    f"🔥 Hold steady... {progress*100:.0f}%",
                    (SCREEN_WIDTH // 2, challenge_y + 165),
                    font_small,
                    COLOR_SUCCESS,
                    center=True,
                )

    elif game.state == GameState.RESULTS:
        draw_text(
            screen,
            "HASIL PERMAINAN",
            (SCREEN_WIDTH // 2, 80),
            font_title,
            COLOR_PRIMARY,
            center=True,
        )
        draw_text(
            screen,
            game.results["mode"],
            (SCREEN_WIDTH // 2, 150),
            font_large,
            COLOR_TEXT,
            center=True,
        )

        # Stats
        card_width = 250
        card_height = 150
        cards = [
            ("SCORE", game.results["score"], COLOR_SUCCESS),
            ("BENAR", f"{game.correct_answers}/{game.results['total']}", COLOR_PRIMARY),
            ("AKURASI", f"{game.results['accuracy']:.1f}%", COLOR_WARNING),
        ]

        spacing = 30
        total_width = len(cards) * card_width + (len(cards) - 1) * spacing
        start_x = (SCREEN_WIDTH - total_width) // 2

        for i, (label, value, color) in enumerate(cards):
            x = start_x + i * (card_width + spacing)
            y = 250
            rect = pygame.Rect(x, y, card_width, card_height)
            draw_card(screen, rect, COLOR_CARD, color, 3)
            draw_text(
                screen,
                label,
                (x + card_width // 2, y + 30),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )
            draw_text(
                screen,
                str(value),
                (x + card_width // 2, y + 90),
                font_large,
                color,
                center=True,
            )

        # Message
        accuracy = game.results["accuracy"]
        if accuracy >= 90:
            message = "🎉 LUAR BIASA!"
            msg_color = COLOR_SUCCESS
        elif accuracy >= 75:
            message = "👍 Bagus sekali!"
            msg_color = COLOR_PRIMARY
        else:
            message = "💪 Terus berlatih!"
            msg_color = COLOR_WARNING

        draw_text(
            screen,
            message,
            (SCREEN_WIDTH // 2, 450),
            font_medium,
            msg_color,
            center=True,
        )

        back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 550, 300, 60)
        draw_button(screen, "KEMBALI", back_rect, font_medium, True)
        draw_text(
            screen,
            "ENTER: Menu utama",
            (SCREEN_WIDTH // 2, 650),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )

    elif game.state == GameState.PAUSED:
        # Overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        card_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT // 2 - 150, 500, 300
        )
        draw_card(screen, card_rect, COLOR_CARD, COLOR_PRIMARY, 3)

        draw_text(
            screen,
            "PAUSE",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80),
            font_title,
            COLOR_PRIMARY,
            center=True,
        )
        draw_text(
            screen,
            "ENTER - Lanjut",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
            font_medium,
            COLOR_TEXT,
            center=True,
        )
        draw_text(
            screen,
            "ESC - Menu",
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50),
            font_medium,
            COLOR_TEXT,
            center=True,
        )

    # Update display
    pygame.display.flip()

# Cleanup
cap.release()
pygame.quit()
print("\n👋 Terima kasih sudah bermain BISINDO BATTLE!")
