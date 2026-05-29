"""
splash.py — Cinématique de démarrage Astralis
─────────────────────────────────────────────
Lance dashboard_orbite.py en arrière-plan,
affiche la cinématique, puis se ferme dès que
le dashboard est prêt (fichier .ready_flag).
La barre bleue reflète la progression réelle
écrite dans .splash_progress.json par le dashboard.
"""
import sys
import os
import json
import subprocess
import random
import math

from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget
from PyQt5.QtCore    import Qt, QTimer, pyqtSignal
from PyQt5.QtGui     import (QPainter, QBrush, QLinearGradient, QRadialGradient,
                              QColor, QFont, QFontMetrics, QPen, QPixmap, QIcon)

# ── Chemins ───────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    APP_DIR     = os.path.dirname(sys.executable)
    _BUNDLE_DIR = getattr(sys, '_MEIPASS', APP_DIR)
else:
    APP_DIR     = os.path.dirname(os.path.abspath(__file__))
    _BUNDLE_DIR = APP_DIR

DASHBOARD   = os.path.join(APP_DIR, 'dashboard_orbite.py')
READY_FLAG  = os.path.join(APP_DIR, '.ready_flag')
PROGRESS_FILE = os.path.join(APP_DIR, '.splash_progress.json')

# ── Nettoyage fichiers résiduels ───────────────────────────────────────────────
for _f in (READY_FLAG, PROGRESS_FILE):
    if os.path.isfile(_f):
        os.remove(_f)


# ── Icône ─────────────────────────────────────────────────────────────────────
def _build_icon() -> QIcon:
    for name in ('astralis_logo.png', 'astralis_logo.jpg', 'logo.png', 'icon.png'):
        for d in (_BUNDLE_DIR, APP_DIR):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                pix = QPixmap(p)
                if not pix.isNull():
                    icon = QIcon()
                    for sz in (16, 32, 48, 64, 128, 256):
                        icon.addPixmap(pix.scaled(sz, sz,
                            Qt.KeepAspectRatioByExpanding,
                            Qt.SmoothTransformation))
                    return icon
    return QIcon()


# ── Palette Control Room ──────────────────────────────────────────────────────
VOID_BG   = "#0B0F19"
VOID_TOP  = "#05070A"
NEON_CYAN = "#00E5FF"
SKY_BLUE  = "#38BDF8"
TEXT_MAIN = "#F8FAFC"
TEXT_MUTE = "#94A3B8"
DEEP_BLUE = "#0F172A"


class SplashWindow(QWidget):

    _LOGO_NAMES = ['astralis_logo.png', 'astralis_logo.jpg', 'logo.png', 'icon.png']

    # Vitesses des phases (ticks à ~55 fps, total ~10 s)
    _SPEED1 = 100 / 120   # fade-in logo   ~2.2 s
    _SPEED2 = 100 / 150   # texte          ~2.7 s
    _SPEED3 = 100 / 180   # barre          ~3.3 s
    # pause finale 1.8 s, puis attente dashboard

    @staticmethod
    def _ease_out(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return 1.0 - (1.0 - t) ** 3

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        screen = QDesktopWidget().screenGeometry()
        self._W = screen.width()
        self._H = screen.height()
        self.resize(self._W, self._H)
        self.move(screen.left(), screen.top())
        self.showFullScreen()

        self._shooting_stars = []
        self._shooting_cooldown = 0

        # Étoiles
        rng = random.Random(42)
        self._stars = [
            (rng.randint(0, self._W), rng.randint(0, self._H),
             rng.uniform(0.5, 2.8),   rng.uniform(0.3, 1.0))
            for _ in range(320)
        ]
        self._twinkling = {rng.randint(0, 319): rng.uniform(0.02, 0.08)
                           for _ in range(40)}
        self._tick_count = 0

        # Phases visuelles (logo + titre) + chargement réel (barre)
        self._phase  = 0.0
        self._phase2 = 0.0
        self._load_progress = 0.0
        self._load_stage = "Lancement du tableau de bord..."
        self._anim_done   = False   # intro logo/titre terminée
        self._dash_ready  = False   # dashboard prêt
        self._closing     = False

        # Logo
        self._logo_pix = None
        logo_size = int(min(self._W, self._H) * 0.28)
        for name in self._LOGO_NAMES:
            for d in (_BUNDLE_DIR, APP_DIR):
                c = os.path.join(d, name)
                if os.path.isfile(c):
                    px = QPixmap(c)
                    if not px.isNull():
                        self._logo_pix = px.scaled(
                            logo_size, logo_size,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    break
            if self._logo_pix:
                break

        # Timer animation (55 fps)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(18)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start()

        # Timer surveillance du flag "dashboard prêt" (toutes les 200 ms)
        self._flag_timer = QTimer(self)
        self._flag_timer.setInterval(200)
        self._flag_timer.timeout.connect(self._check_ready)
        self._flag_timer.start()

        # Lancement du dashboard en arrière-plan
        self._launch_dashboard()

    # ── Lancement dashboard ───────────────────────────────
    def _launch_dashboard(self):
        python = sys.executable
        try:
            subprocess.Popen(
                [python, DASHBOARD, '--splash'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except Exception as e:
            print(f"[splash] Erreur lancement dashboard : {e}")

    # ── Surveillance flag / progression ───────────────────
    def _read_load_progress(self):
        """Lit la progression réelle écrite par dashboard_orbite.py."""
        if not os.path.isfile(PROGRESS_FILE):
            return
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._load_progress = float(data.get("progress", 0))
            stage = (data.get("stage") or "").strip()
            if stage:
                self._load_stage = stage
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    def _schedule_close(self, delay_ms: int = 300):
        if self._closing:
            return
        if self._dash_ready and (self._phase2 >= 70.0 or self._anim_done):
            self._closing = True
            QTimer.singleShot(delay_ms, self._close)

    def _check_ready(self):
        self._read_load_progress()
        if self._load_progress >= 100.0 or os.path.isfile(READY_FLAG):
            self._dash_ready = True
            self._load_progress = max(self._load_progress, 100.0)
            self._flag_timer.stop()
            self._schedule_close()

    # ── Tick animation ────────────────────────────────────
    def _tick(self):
        self._tick_count += 1
        self._read_load_progress()
        if self._phase < 100:
            self._phase = min(100.0, self._phase + self._SPEED1)
        elif self._phase2 < 100:
            self._phase2 = min(100.0, self._phase2 + self._SPEED2)
        elif not self._anim_done:
            self._anim_done = True
        self._schedule_close()
        if self._shooting_cooldown > 0:
            self._shooting_cooldown -= 1
        elif random.random() < 0.018:
            angle = random.uniform(-0.6, -0.15)
            speed = random.uniform(14, 28)
            self._shooting_stars.append([
                random.uniform(0, self._W * 0.7),
                random.uniform(0, self._H * 0.45),
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                random.uniform(40, 90),
            ])
            self._shooting_cooldown = random.randint(25, 70)
        alive = []
        for star in self._shooting_stars:
            star[0] += star[2]
            star[1] += star[3]
            star[4] -= 1.2
            if star[4] > 0 and star[0] < self._W + 80 and star[1] < self._H + 80:
                alive.append(star)
        self._shooting_stars = alive
        self.update()

    def _close(self):
        self._anim_timer.stop()
        self._flag_timer.stop()
        for path in (READY_FLAG, PROGRESS_FILE):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        self.close()
        QApplication.instance().quit()

    # ── Clic pour passer ──────────────────────────────────
    def mousePressEvent(self, _event):
        if not self._dash_ready:
            self._phase = self._phase2 = 100.0
            self._anim_done = True
        else:
            self._schedule_close(100)

    # ── Rendu ─────────────────────────────────────────────
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        W, H = self._W, self._H

        # ── Layout vertical calculé ──────────────────────
        font_sz  = max(28, min(72, int(W / 20)))
        sub_sz   = max(11, min(22, int(W / 80)))
        load_sz  = max(9,  min(16, int(W / 120)))
        hint_sz  = max(8,  min(13, int(W / 150)))
        logo_size = int(min(W, H) * 0.28)
        bar_thick = max(4, int(H * 0.006))
        bar_w     = int(W * 0.45)

        font_title = QFont("Segoe UI", font_sz, QFont.Light)
        font_title.setLetterSpacing(QFont.PercentageSpacing, 155)
        fm_t   = QFontMetrics(font_title)
        title_h = fm_t.height()
        title_w = fm_t.horizontalAdvance("ASTRALIS")

        font_sub = QFont("Segoe UI", sub_sz, QFont.Light)
        font_sub.setLetterSpacing(QFont.PercentageSpacing, 125)
        sub_h = QFontMetrics(font_sub).height()

        font_load = QFont("Segoe UI", load_sz, QFont.Light)
        label_h = QFontMetrics(font_load).height()

        GAP1 = int(H * 0.040)
        GAP2 = int(H * 0.018)
        GAP3 = int(H * 0.018)
        GAP4 = int(H * 0.055)
        GAP5 = int(H * 0.015)
        MARGE_BOT = int(H * 0.060)

        bloc_h = (logo_size + GAP1 + title_h + GAP2 + 2 + GAP3 +
                  sub_h + GAP4 + bar_thick + GAP5 + label_h + MARGE_BOT)
        y0 = max(int(H * 0.05), (H - bloc_h) // 2)

        y_logo  = y0
        y_title = y_logo  + logo_size + GAP1
        y_line  = y_title + title_h   + GAP2
        y_sub   = y_line  + 2         + GAP3
        y_bar   = y_sub   + sub_h     + GAP4
        y_label = y_bar   + bar_thick + GAP5
        y_hint  = H - int(MARGE_BOT * 0.4)
        cx = W // 2

        # ── Fond — vide spatial ─────────────────────────
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0.0,  QColor(VOID_TOP))
        bg.setColorAt(0.45, QColor(VOID_BG))
        bg.setColorAt(1.0,  QColor(VOID_BG))
        p.fillRect(0, 0, W, H, QBrush(bg))

        # Nébuleuses cyan / bleu profond
        neb_a = QRadialGradient(W * 0.22, H * 0.18, W * 0.55)
        neb_a.setColorAt(0.0, QColor(0, 229, 255, 22))
        neb_a.setColorAt(0.35, QColor(15, 40, 80, 14))
        neb_a.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(neb_a))
        p.setPen(Qt.NoPen)
        p.drawRect(0, 0, W, H)

        neb_b = QRadialGradient(W * 0.82, H * 0.72, W * 0.48)
        neb_b.setColorAt(0.0, QColor(56, 189, 248, 16))
        neb_b.setColorAt(0.4, QColor(10, 25, 55, 12))
        neb_b.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(neb_b))
        p.drawRect(0, 0, W, H)

        neb_c = QRadialGradient(W * 0.5, H * 0.55, max(W, H) * 0.35)
        neb_c.setColorAt(0.0, QColor(8, 20, 45, 30))
        neb_c.setColorAt(0.6, QColor(11, 15, 25, 10))
        neb_c.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(neb_c))
        p.drawRect(0, 0, W, H)

        # Vignette
        vig = QRadialGradient(W / 2, H / 2, max(W, H) * 0.72)
        vig.setColorAt(0.0, QColor(0, 0, 0, 0))
        vig.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.setBrush(QBrush(vig))
        p.drawRect(0, 0, W, H)

        # ── Étoiles (tons cyan / bleu) ──────────────────
        star_appear = min(1.0, self._tick_count / 55)
        for i, (sx, sy, sr, sa) in enumerate(self._stars):
            if i in self._twinkling:
                flicker = 0.5 + 0.5 * math.sin(self._tick_count * self._twinkling[i])
                alpha = int(sa * 220 * star_appear * (0.3 + 0.7 * flicker))
            else:
                alpha = int(sa * 200 * star_appear)
            if i % 4 == 0:
                star_col = QColor(0, 229, 255, alpha)
            elif i % 4 == 1:
                star_col = QColor(56, 189, 248, alpha)
            elif i % 4 == 2:
                star_col = QColor(120, 200, 255, alpha)
            else:
                star_col = QColor(200, 230, 255, int(alpha * 0.85))
            p.setBrush(QBrush(star_col))
            p.setPen(Qt.NoPen)
            p.drawEllipse(int(sx - sr), int(sy - sr), int(sr * 2), int(sr * 2))

        # ── Points lumineux en mouvement ─────────────────
        for sx, sy, _, _, life in self._shooting_stars:
            alpha = int(min(255, life * 2.8))
            size = max(2, int(2 + life * 0.04))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 229, 255, alpha)))
            p.drawEllipse(int(sx - size / 2), int(sy - size / 2), size, size)

        # ── Halo logo ────────────────────────────────────
        cy_logo = y_logo + logo_size // 2
        halo_r  = int(logo_size * 0.75)
        halo = QRadialGradient(cx, cy_logo, halo_r)
        ha = int((self._phase / 100) * 60)
        halo.setColorAt(0.0, QColor(0,   229, 255, ha))
        halo.setColorAt(0.4, QColor(56,  189, 248, ha // 2))
        halo.setColorAt(1.0, QColor(8,   30,  60,  0))
        p.setBrush(QBrush(halo))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx - halo_r), int(cy_logo - halo_r),
                      int(halo_r * 2), int(halo_r * 2))

        # ── Logo ─────────────────────────────────────────
        p.setOpacity(self._phase / 100.0)
        if self._logo_pix and not self._logo_pix.isNull():
            lw, lh2 = self._logo_pix.width(), self._logo_pix.height()
            p.drawPixmap(cx - lw // 2, y_logo + (logo_size - lh2) // 2, self._logo_pix)
        p.setOpacity(1.0)

        # ── Titre ────────────────────────────────────────
        if self._phase2 > 0:
            fade = self._ease_out(self._phase2 / 100.0)
            txt_alpha = int(fade * 255)
            p.setFont(font_title)
            if txt_alpha >= 2:
                glow_a = int(txt_alpha * 0.22)
                p.setPen(QColor(0, 229, 255, glow_a))
                p.drawText(0, y_title + 1, W, title_h + 4, Qt.AlignHCenter, "ASTRALIS")
                p.setPen(QColor(248, 250, 252, txt_alpha))
                p.drawText(0, y_title, W, title_h + 4, Qt.AlignHCenter, "ASTRALIS")

        # ── Ligne décorative ─────────────────────────────
        if self._phase2 > 20:
            prog = self._ease_out(min(1.0, (self._phase2 - 20) / 55))
            lhalf = int(title_w * 0.42 * prog)
            la = int(prog * 130)
            if lhalf > 2:
                gl = QLinearGradient(cx - lhalf, 0, cx + lhalf, 0)
                gl.setColorAt(0.0, QColor(0,   229, 255, 0))
                gl.setColorAt(0.5, QColor(0,   229, 255, la))
                gl.setColorAt(1.0, QColor(0,   229, 255, 0))
                p.setBrush(QBrush(gl))
                p.setPen(Qt.NoPen)
                p.drawRect(cx - lhalf, y_line, lhalf * 2, 1)

        # ── Sous-titre ────────────────────────────────────
        if self._phase2 > 8:
            sub_fade = self._ease_out(min(1.0, (self._phase2 - 8) / 92))
            sub_alpha = int(sub_fade * 255)
            p.setFont(font_sub)
            p.setPen(QColor(148, 163, 184, sub_alpha))
            p.drawText(0, y_sub, W, sub_h + 4,
                       Qt.AlignHCenter, "Moteur de simulation orbitale N-corps")

        # ── Barre (progression réelle du dashboard) ─────
        if self._load_progress > 0:
            progress = self._load_progress / 100.0
            bar_label = self._load_stage
        elif self._phase2 > 40:
            progress = min(0.12, self._phase2 / 100.0 * 0.12)
            bar_label = "Lancement du tableau de bord..."
        else:
            progress = 0.0
            bar_label = "Initialisation..."

        bar_x = (W - bar_w) // 2
        p.setBrush(QBrush(QColor(26, 35, 58, 120)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, y_bar, bar_w, bar_thick,
                          bar_thick // 2, bar_thick // 2)
        fill = int(bar_w * progress)
        if fill > 2:
            gb = QLinearGradient(bar_x, 0, bar_x + fill, 0)
            gb.setColorAt(0.0, QColor(8,   30,  55,  220))
            gb.setColorAt(0.45, QColor(56,  189, 248, 240))
            gb.setColorAt(1.0, QColor(0,   229, 255, 255))
            p.setBrush(QBrush(gb))
            p.drawRoundedRect(bar_x, y_bar, fill, bar_thick,
                              bar_thick // 2, bar_thick // 2)
            tip   = bar_x + fill
            gw_r  = bar_thick * 5.5
            gw    = QRadialGradient(tip, y_bar + bar_thick / 2, gw_r)
            gw.setColorAt(0.0, QColor(0,   229, 255, 255))
            gw.setColorAt(0.35, QColor(0,   229, 255, 140))
            gw.setColorAt(0.7, QColor(56,  189, 248, 50))
            gw.setColorAt(1.0, QColor(0,   180, 220, 0))
            p.setBrush(QBrush(gw))
            p.drawEllipse(int(tip - gw_r), int(y_bar + bar_thick / 2 - gw_r),
                          int(gw_r * 2), int(gw_r * 2))
            p.setBrush(QBrush(QColor(248, 250, 252, 220)))
            p.drawEllipse(int(tip - 2), int(y_bar + bar_thick / 2 - 2), 4, 4)

        # Label barre
        if progress > 0.01 or self._phase2 > 25:
            la2 = int(min(1.0, max(progress, self._phase2 / 100.0)) * 120)
            p.setFont(font_load)
            p.setPen(QColor(0, 229, 255, la2))
            pct_txt = f" — {int(self._load_progress)} %" if self._load_progress > 0 else ""
            p.drawText(0, y_label, W, label_h + 4, Qt.AlignHCenter, bar_label + pct_txt)

        # Hint clic
        if self._phase2 > 30 and not self._anim_done:
            ha2 = int(min(1.0, (self._phase2 - 30) / 40) * 60)
            fh = QFont("Segoe UI", hint_sz, QFont.Light)
            p.setFont(fh)
            p.setPen(QColor(148, 163, 184, ha2))
            p.drawText(0, y_hint, W, hint_sz * 2 + 4,
                       Qt.AlignHCenter, "Appuyer pour passer")

        p.end()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(_build_icon())

    win = SplashWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
