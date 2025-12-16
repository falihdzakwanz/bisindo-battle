"""
BISINDO BATTLE - Game Rendering & UI Components
Complete rendering system untuk semua game states
"""

import pygame
import cv2
import numpy as np

# Import dari main game file
from bisindo_game import *


def render_webcam_feed(
    screen, frame, pos, size, show_landmarks=False, landmarks=None, bbox=None
):
    """Render webcam feed ke pygame surface dengan optional landmarks"""
    if frame is None:
        return

    # Resize frame
    frame_resized = cv2.resize(frame, size)

    # Draw landmarks jika debug mode
    if show_landmarks and landmarks and bbox:
        x_min, y_min, x_max, y_max = bbox

        # Draw bounding box
        cv2.rectangle(
            frame_resized,
            (
                int(x_min * size[0] / frame.shape[1]),
                int(y_min * size[1] / frame.shape[0]),
            ),
            (
                int(x_max * size[0] / frame.shape[1]),
                int(y_max * size[1] / frame.shape[0]),
            ),
            (0, 255, 0),
            2,
        )

        # Draw landmarks
        h, w = size[1], size[0]
        for lm in landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_resized, (cx, cy), 3, (255, 0, 0), -1)

        # Draw connections
        mp_drawing.draw_landmarks(
            frame_resized,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # Convert to pygame surface
    frame_surf = pygame.surfarray.make_surface(np.rot90(frame_resized))
    screen.blit(frame_surf, pos)

    # Draw border
    border_rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
    pygame.draw.rect(screen, COLOR_TEXT_DIM, border_rect, 2, border_radius=10)


def render_main_menu(screen, game):
    """Render main menu"""
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
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )
    draw_text(
        screen,
        "Tekan D untuk toggle debug mode",
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 20),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )


def render_mode_select(screen, game):
    """Render mode selection screen"""
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
        (
            "Challenge Mode",
            "Tunjukkan 10 huruf secara berurutan",
            "🎯 Fixed rounds, no time limit",
        ),
        ("Practice Mode", "Latihan bebas selama 3 menit", "🎓 Learn at your own pace"),
        ("Time Attack", "Score maksimal dalam 60 detik!", "⚡ Fast-paced, competitive"),
    ]

    card_width = 350
    card_height = 200
    spacing = 30
    start_x = (SCREEN_WIDTH - (card_width * 3 + spacing * 2)) // 2

    for i, (mode, desc, detail) in enumerate(mode_descriptions):
        x = start_x + i * (card_width + spacing)
        y = 250

        rect = pygame.Rect(x, y, card_width, card_height)

        # Card background
        color = COLOR_PRIMARY if i == game.mode_index else COLOR_CARD
        draw_card(
            screen,
            rect,
            color,
            COLOR_PRIMARY if i == game.mode_index else None,
            3 if i == game.mode_index else 0,
        )

        # Mode title
        draw_text(
            screen,
            mode,
            (x + card_width // 2, y + 40),
            font_medium,
            COLOR_TEXT,
            center=True,
        )

        # Description
        lines = [desc, "", detail]
        for j, line in enumerate(lines):
            draw_text(
                screen,
                line,
                (x + card_width // 2, y + 90 + j * 30),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )

    # Instructions
    draw_text(
        screen,
        "Gunakan ↑↓ atau ←→ untuk memilih mode",
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )
    draw_text(
        screen,
        "ENTER untuk mulai | ESC untuk kembali",
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )


def render_challenge_mode(
    screen,
    game,
    last_frame,
    last_prediction,
    last_confidence,
    hand_detected,
    landmarks_vis,
    bbox,
):
    """Render challenge mode gameplay"""
    # Left panel - Webcam
    webcam_size = (480, 360)
    webcam_pos = (50, 120)
    render_webcam_feed(
        screen,
        last_frame,
        webcam_pos,
        webcam_size,
        game.debug_mode,
        landmarks_vis,
        bbox,
    )

    # Debug info
    if game.debug_mode:
        debug_y = webcam_pos[1] + webcam_size[1] + 20
        draw_text(
            screen,
            f"Hand Detected: {'YES' if hand_detected else 'NO'}",
            (webcam_pos[0], debug_y),
            font_small,
            COLOR_SUCCESS if hand_detected else COLOR_ERROR,
        )
        draw_text(
            screen,
            f"Landmarks: {len(landmarks_vis.landmark) if landmarks_vis else 0} points",
            (webcam_pos[0], debug_y + 25),
            font_small,
            COLOR_TEXT_DIM,
        )

    # Right panel - Game info
    right_panel_x = 600

    # Header
    draw_text(screen, "CHALLENGE MODE", (right_panel_x, 50), font_large, COLOR_PRIMARY)

    # Round info
    round_text = f"Round {game.round} / {game.total_rounds}"
    draw_text(screen, round_text, (right_panel_x, 100), font_medium, COLOR_TEXT)

    # Score
    score_rect = pygame.Rect(right_panel_x, 150, 300, 80)
    draw_card(screen, score_rect, COLOR_CARD, COLOR_PRIMARY, 2)
    draw_text(
        screen,
        "SCORE",
        (right_panel_x + 150, 165),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )
    draw_text(
        screen,
        str(game.score),
        (right_panel_x + 150, 200),
        font_large,
        COLOR_SUCCESS,
        center=True,
    )

    # Target letter (big display)
    target_rect = pygame.Rect(right_panel_x, 260, 300, 200)
    draw_card(screen, target_rect, COLOR_PRIMARY)
    draw_text(
        screen,
        "TUNJUKKAN HURUF:",
        (right_panel_x + 150, 280),
        font_small,
        COLOR_TEXT,
        center=True,
    )
    draw_text(
        screen,
        game.current_letter,
        (right_panel_x + 150, 360),
        font_title,
        COLOR_TEXT,
        center=True,
    )

    # Prediction display
    pred_rect = pygame.Rect(right_panel_x, 490, 300, 120)

    if hand_detected and last_prediction:
        # Color based on match
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
            (right_panel_x + 150, 505),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )
        draw_text(
            screen,
            last_prediction,
            (right_panel_x + 150, 545),
            font_large,
            pred_color,
            center=True,
        )

        # Confidence bar
        conf_rect = pygame.Rect(right_panel_x + 30, 590, 240, 15)
        draw_progress_bar(screen, conf_rect, last_confidence, pred_color)
        draw_text(
            screen,
            f"{last_confidence*100:.1f}%",
            (right_panel_x + 150, 605),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )
    else:
        draw_card(screen, pred_rect, COLOR_CARD, COLOR_TEXT_DIM, 2)
        draw_text(
            screen,
            "Tidak ada tangan terdeteksi",
            (right_panel_x + 150, 545),
            font_small,
            COLOR_TEXT_DIM,
            center=True,
        )

    # Instructions
    instruction_y = 630
    draw_text(
        screen, "INSTRUKSI:", (right_panel_x, instruction_y), font_small, COLOR_TEXT
    )
    draw_text(
        screen,
        "• Tunjukkan gesture huruf di atas",
        (right_panel_x, instruction_y + 25),
        font_small,
        COLOR_TEXT_DIM,
    )
    draw_text(
        screen,
        "• Tekan SPACE untuk submit",
        (right_panel_x, instruction_y + 50),
        font_small,
        COLOR_TEXT_DIM,
    )
    draw_text(
        screen,
        "• ESC untuk pause",
        (right_panel_x, instruction_y + 75),
        font_small,
        COLOR_TEXT_DIM,
    )


def render_practice_mode(
    screen,
    game,
    last_frame,
    last_prediction,
    last_confidence,
    hand_detected,
    landmarks_vis,
    bbox,
):
    """Render practice mode"""
    # Similar to challenge but with timer and different layout
    elapsed = time.time() - game.start_time
    remaining = max(0, game.time_limit - elapsed)

    # Top bar - Timer
    timer_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 20, 300, 60)
    draw_card(screen, timer_rect, COLOR_CARD, COLOR_PRIMARY, 2)

    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    draw_text(
        screen,
        timer_text,
        (SCREEN_WIDTH // 2, 50),
        font_large,
        COLOR_PRIMARY,
        center=True,
    )

    # Webcam (center)
    webcam_size = (640, 480)
    webcam_pos = ((SCREEN_WIDTH - webcam_size[0]) // 2, 100)
    render_webcam_feed(
        screen,
        last_frame,
        webcam_pos,
        webcam_size,
        game.debug_mode,
        landmarks_vis,
        bbox,
    )

    # Bottom info panel
    info_y = webcam_pos[1] + webcam_size[1] + 20

    # Stats row
    stats_width = 200
    stats = [
        ("Benar", game.correct_answers, COLOR_SUCCESS),
        ("Attempts", game.attempts, COLOR_TEXT),
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

    # Current challenge
    challenge_y = info_y + 90
    draw_text(
        screen,
        "TUNJUKKAN HURUF:",
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
        pred_text = f"Prediksi: {last_prediction} ({last_confidence*100:.0f}%)"
        pred_color = (
            COLOR_SUCCESS
            if last_prediction == game.current_letter
            and last_confidence >= CONFIDENCE_THRESHOLD
            else COLOR_TEXT_DIM
        )
        draw_text(
            screen,
            pred_text,
            (SCREEN_WIDTH // 2, challenge_y + 110),
            font_small,
            pred_color,
            center=True,
        )


def render_time_attack(
    screen,
    game,
    last_frame,
    last_prediction,
    last_confidence,
    hand_detected,
    landmarks_vis,
    bbox,
):
    """Render time attack mode - fast-paced gameplay"""
    elapsed = time.time() - game.start_time
    remaining = max(0, game.time_limit - elapsed)

    # Progress bar at top
    progress = elapsed / game.time_limit
    progress_rect = pygame.Rect(50, 20, SCREEN_WIDTH - 100, 40)
    draw_progress_bar(
        screen,
        progress_rect,
        progress,
        COLOR_ERROR if remaining < 10 else COLOR_PRIMARY,
    )

    timer_text = f"{int(remaining)}s"
    draw_text(
        screen,
        timer_text,
        (SCREEN_WIDTH // 2, 40),
        font_large,
        COLOR_ERROR if remaining < 10 else COLOR_TEXT,
        center=True,
    )

    # Split screen layout
    # Left - Webcam
    webcam_size = (500, 375)
    webcam_pos = (50, 90)
    render_webcam_feed(
        screen,
        last_frame,
        webcam_pos,
        webcam_size,
        game.debug_mode,
        landmarks_vis,
        bbox,
    )

    # Right - Game info (compact)
    right_x = 600

    # Target (huge)
    target_rect = pygame.Rect(right_x, 90, 600, 250)
    draw_card(screen, target_rect, COLOR_PRIMARY)
    draw_text(
        screen,
        game.current_letter,
        (right_x + 300, 215),
        pygame.font.Font(None, 180),
        COLOR_TEXT,
        center=True,
    )

    # Score & streak
    score_y = 370
    draw_text(
        screen,
        f"SCORE: {game.score}",
        (right_x + 300, score_y),
        font_large,
        COLOR_SUCCESS,
        center=True,
    )
    draw_text(
        screen,
        f"Benar: {game.correct_answers} / {game.attempts}",
        (right_x + 300, score_y + 50),
        font_medium,
        COLOR_TEXT_DIM,
        center=True,
    )

    # Prediction feedback (instant)
    if hand_detected and last_prediction:
        feedback_y = 480
        is_match = last_prediction == game.current_letter

        if is_match and last_confidence >= CONFIDENCE_THRESHOLD:
            draw_text(
                screen,
                "✓ BENAR!",
                (right_x + 300, feedback_y),
                font_large,
                COLOR_SUCCESS,
                center=True,
            )
            draw_text(
                screen,
                "Tekan SPACE cepat!",
                (right_x + 300, feedback_y + 50),
                font_medium,
                COLOR_TEXT,
                center=True,
            )
        else:
            draw_text(
                screen,
                f"Prediksi: {last_prediction}",
                (right_x + 300, feedback_y),
                font_medium,
                COLOR_TEXT_DIM,
                center=True,
            )
            draw_text(
                screen,
                f"Confidence: {last_confidence*100:.0f}%",
                (right_x + 300, feedback_y + 40),
                font_small,
                COLOR_TEXT_DIM,
                center=True,
            )


def render_results(screen, game):
    """Render results screen"""
    # Title
    draw_text(
        screen,
        "HASIL PERMAINAN",
        (SCREEN_WIDTH // 2, 80),
        font_title,
        COLOR_PRIMARY,
        center=True,
    )

    # Mode
    draw_text(
        screen,
        game.results["mode"],
        (SCREEN_WIDTH // 2, 150),
        font_large,
        COLOR_TEXT,
        center=True,
    )

    # Stats cards
    card_width = 250
    card_height = 150
    cards = [
        ("SCORE", game.results["score"], COLOR_SUCCESS),
        ("BENAR", f"{game.results['total']}", COLOR_PRIMARY),
        ("AKURASI", f"{game.results['accuracy']:.1f}%", COLOR_WARNING),
    ]

    if game.results["time"] > 0:
        cards.append(("WAKTU", f"{game.results['time']:.1f}s", COLOR_TEXT))

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

    # Performance message
    accuracy = game.results["accuracy"]
    if accuracy >= 90:
        message = "🎉 LUAR BIASA! Perfect score!"
        msg_color = COLOR_SUCCESS
    elif accuracy >= 75:
        message = "👍 Bagus! Terus berlatih!"
        msg_color = COLOR_PRIMARY
    elif accuracy >= 50:
        message = "💪 Cukup baik, masih bisa ditingkatkan!"
        msg_color = COLOR_WARNING
    else:
        message = "📚 Jangan menyerah, terus belajar!"
        msg_color = COLOR_TEXT_DIM

    draw_text(
        screen, message, (SCREEN_WIDTH // 2, 450), font_medium, msg_color, center=True
    )

    # Back button
    back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 550, 300, 60)
    draw_button(screen, "KEMBALI KE MENU", back_rect, font_medium, True)

    draw_text(
        screen,
        "Tekan ENTER untuk kembali",
        (SCREEN_WIDTH // 2, 650),
        font_small,
        COLOR_TEXT_DIM,
        center=True,
    )


def render_paused(screen, game):
    """Render pause overlay"""
    # Semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    # Pause card
    card_rect = pygame.Rect(SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT // 2 - 150, 500, 300)
    draw_card(screen, card_rect, COLOR_CARD, COLOR_PRIMARY, 3)

    draw_text(
        screen,
        "PAUSE",
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80),
        font_title,
        COLOR_PRIMARY,
        center=True,
    )

    options = ["ENTER - Lanjutkan", "ESC - Kembali ke menu"]

    for i, option in enumerate(options):
        draw_text(
            screen,
            option,
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 50),
            font_medium,
            COLOR_TEXT,
            center=True,
        )
