"""Ventana principal (PySide6, sin pantalla): comportamiento del layout con ventanas pequeñas.

- Panel de controles: si la ventana es baja, aparece una barra de desplazamiento y los botones conservan
  su alto (no se aplastan unos sobre otros); con espacio de sobra, la barra desaparece.
- Tablas de abajo: arrastrar con el clic izquierdo desplaza (QScroller); un clic corto selecciona la fila.
- Visor (VideoView): zoom sobre el cursor, paneo sin salirse de la imagen, clic corregido por el zoom y
  arrastrar sin elegir moto; pantalla completa con F11 / Esc.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.synthetic import test_config  # noqa: E402

try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QPushButton, QTableWidgetItem, QTabWidget
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class MainWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self, width: int, height: int):
        from ui.main_window import MainWindow
        win = MainWindow(test_config())
        self.addCleanup(win.deleteLater)
        win.resize(width, height)
        win.show()
        QTest.qWaitForWindowExposed(win)
        QTest.qWait(50)
        return win

    def _control_buttons(self, win) -> list[QPushButton]:
        return win.controls_scroll.widget().findChildren(QPushButton)

    def test_short_window_scrolls_controls_instead_of_squashing(self):
        win = self._window(1200, 420)
        bar = win.controls_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0, "con la ventana baja debe aparecer la barra de desplazamiento")
        for b in self._control_buttons(win):
            self.assertGreaterEqual(b.height(), b.minimumSizeHint().height(), f"botón aplastado: {b.text()!r}")

    def test_stop_button_glows_only_while_enabled(self):
        win = self._window(1600, 980)
        self.assertEqual(win.btn_stop.objectName(), "dangerButton")
        self.assertFalse(win.btn_stop.isEnabled())
        self.assertFalse(win.btn_stop.glow.isEnabled(), "en reposo DETENER no brilla")
        win.btn_stop.setEnabled(True)
        self.assertTrue(win.btn_stop.glow.isEnabled(), "mientras hay algo que detener, brilla en rojo")

    def test_panels_are_rounded_but_data_tables_stay_square(self):
        import re
        from ui.theme import STYLE
        rules = dict(re.findall(r"^([^{\n]+?)\s*\{([^}]*)\}", STYLE, flags=re.M))

        def radius(selector_start):
            body = next(b for sel, b in rules.items() if sel.startswith(selector_start))
            m = re.search(r"border-radius:\s*(\d+)px", body)
            return int(m.group(1)) if m else 0

        for sel in ("QFrame#card", "QPushButton {", "QLabel#chip", "QLineEdit", "QTextBrowser"):
            self.assertGreater(radius(sel.rstrip(" {")), 0, f"{sel} debe tener esquinas redondeadas")
        self.assertEqual(radius("QTableWidget"), 0, "la tabla de datos de las motos queda con esquinas rectas")

    def test_tall_window_needs_no_scroll(self):
        win = self._window(1600, 1600)
        self.assertEqual(win.controls_scroll.verticalScrollBar().maximum(), 0)


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class TableDragScrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui.main_window import MainWindow
        self.win = MainWindow(test_config())
        self.addCleanup(self.win.deleteLater)
        self.table = self.win.events_table                  # sin manejador de selección: solo el gesto
        self.table.setRowCount(200)
        for r in range(200):
            self.table.setItem(r, 0, QTableWidgetItem(str(r)))
        # QScroller reenvía el clic al widget bajo el cursor (QApplication.widgetAt): la tabla tiene que quedar
        # dentro de la pantalla virtual (offscreen), si no el clic no llega a ningún lado
        screen = self.win.screen().availableGeometry()
        self.win.setGeometry(screen)
        self.win.show()
        QTest.qWaitForWindowExposed(self.win)
        self._show_events_tab()

    def _show_events_tab(self):
        tabs = self.table.parentWidget()
        while not isinstance(tabs, QTabWidget):
            tabs = tabs.parentWidget()
        tabs.setCurrentWidget(self.table)
        QTest.qWait(50)

    def _selected_rows(self) -> list[int]:
        return [i.row() for i in self.table.selectionModel().selectedRows()]

    def test_click_selects_row(self):
        vp = self.table.viewport()
        row_y = self.table.rowViewportPosition(1) + self.table.rowHeight(1) // 2
        QTest.mouseClick(vp, Qt.MouseButton.LeftButton, pos=QPoint(20, row_y))
        QTest.qWait(300)
        self.assertEqual(self._selected_rows(), [1])

    def test_left_drag_scrolls_without_changing_selection(self):
        vp = self.table.viewport()
        bottom = vp.height() - 10
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, pos=QPoint(20, bottom))
        QTest.qWait(20)
        for y in range(bottom, 10, -10):
            QTest.mouseMove(vp, QPoint(20, y))
            QTest.qWait(15)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=QPoint(20, 10))
        QTest.qWait(800)
        self.assertGreater(self.table.verticalScrollBar().value(), 0, "el arrastre debe desplazar la tabla")
        self.assertEqual(self._selected_rows(), [], "el arrastre no debe seleccionar filas")


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class VideoViewZoomTests(unittest.TestCase):
    """Zoom y paneo del visor: los mismos para cámara, video y foto (todos llegan como set_image)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtGui import QImage
        from ui.widgets import VideoView
        self.view = VideoView()
        self.addCleanup(self.view.deleteLater)
        self.view.resize(800, 450)                           # 2x una imagen de 400x225: escala exacta
        img = QImage(400, 225, QImage.Format.Format_RGB888)
        img.fill(0)
        self.view.set_image(img)
        self.view.show()
        QTest.qWaitForWindowExposed(self.view)
        self.clicks = []
        self.view.clicked.connect(lambda x, y: self.clicks.append((x, y)))

    def _center_px(self):
        return QPoint(self.view.width() // 2, self.view.height() // 2)

    def test_click_without_zoom_maps_to_frame_pixels(self):
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(200, 100))
        self.assertEqual(self.clicks, [(100.0, 50.0)])

    def test_zoom_keeps_the_point_under_the_cursor(self):
        from PySide6.QtCore import QPointF
        anchor = QPointF(200, 100)
        before = self.view.widget_to_image(anchor)
        self.view.zoom_by(2.0, anchor)
        self.assertAlmostEqual(self.view.zoom, 2.0)
        after = self.view.widget_to_image(anchor)
        self.assertAlmostEqual(after.x(), before.x(), places=6)
        self.assertAlmostEqual(after.y(), before.y(), places=6)

    def test_click_after_zoom_is_corrected(self):
        self.view.zoom_by(2.0)                              # centrado: el centro del visor sigue en (200, 112.5)
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=self._center_px())
        x, y = self.clicks[0]
        self.assertAlmostEqual(x, 200, delta=0.5)
        self.assertAlmostEqual(y, 112.5, delta=0.5)
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(200, 100))   # 4x: (400-200)/4 = 50 px
        x, y = self.clicks[1]
        self.assertAlmostEqual(x, 200 - (400 - 200) / 4, delta=0.5)

    def test_drag_pans_and_does_not_select(self):
        self.view.zoom_by(2.0)
        center_before = self.view.widget_to_image(QPoint(400, 225).toPointF())
        QTest.mousePress(self.view, Qt.MouseButton.LeftButton, pos=QPoint(400, 225))
        QTest.mouseMove(self.view, QPoint(350, 225))
        QTest.mouseMove(self.view, QPoint(300, 225))
        QTest.mouseRelease(self.view, Qt.MouseButton.LeftButton, pos=QPoint(300, 225))
        center_after = self.view.widget_to_image(QPoint(400, 225).toPointF())
        self.assertAlmostEqual(center_after.x() - center_before.x(), 100 / 4, delta=0.5)  # 100 px / escala 4
        self.assertEqual(self.clicks, [], "arrastrar para mover no debe elegir una moto")

    def test_image_edges_cannot_be_dragged_into_view(self):
        self.view.zoom_by(2.0)
        for _ in range(2):                                  # arrastra hacia abajo a la derecha más de lo posible
            QTest.mousePress(self.view, Qt.MouseButton.MiddleButton, pos=QPoint(10, 10))
            QTest.mouseMove(self.view, QPoint(790, 440))
            QTest.mouseRelease(self.view, Qt.MouseButton.MiddleButton, pos=QPoint(790, 440))
        top_left = self.view.widget_to_image(QPoint(0, 0).toPointF())
        self.assertAlmostEqual(top_left.x(), 0, delta=0.01)
        self.assertAlmostEqual(top_left.y(), 0, delta=0.01)

    def test_zoom_survives_new_frames_of_the_same_size_and_resets_otherwise(self):
        from PySide6.QtGui import QImage
        self.view.zoom_by(2.0)
        same = QImage(400, 225, QImage.Format.Format_RGB888)
        self.view.set_image(same)                           # siguiente frame del video en vivo
        self.assertAlmostEqual(self.view.zoom, 2.0)
        self.view.set_image(QImage(640, 480, QImage.Format.Format_RGB888))   # otra foto, otro tamaño
        self.assertAlmostEqual(self.view.zoom, 1.0)

    def test_keys_zoom_and_reset_and_arrows_pass_through(self):
        QTest.keyClick(self.view, Qt.Key.Key_Plus)
        self.assertGreater(self.view.zoom, 1.0)
        QTest.keyClick(self.view, Qt.Key.Key_0)
        self.assertEqual(self.view.zoom, 1.0)
        QTest.mouseDClick(self.view, Qt.MouseButton.LeftButton, pos=self._center_px())
        self.assertEqual(self.view.zoom, 1.0)
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
        self.view.keyPressEvent(ev)
        self.assertFalse(ev.isAccepted(), "← → deben seguir hacia la ventana (modo foto)")


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class VideoFullscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui.main_window import MainWindow
        self.win = MainWindow(test_config())
        self.addCleanup(self.win.deleteLater)
        self.win.show()
        QTest.qWaitForWindowExposed(self.win)

    def test_f11_shows_only_the_viewer_and_esc_restores(self):
        chrome = self.win._fullscreen_hidden_widgets()
        QTest.keyClick(self.win, Qt.Key.Key_F11)
        self.assertTrue(self.win.video_fullscreen)
        self.assertTrue(self.win.isFullScreen())
        self.assertTrue(all(w.isHidden() for w in chrome))
        self.assertTrue(self.win.video.isVisible())
        QTest.keyClick(self.win, Qt.Key.Key_Escape)
        self.assertFalse(self.win.video_fullscreen)
        self.assertFalse(self.win.isFullScreen())
        self.assertTrue(all(not w.isHidden() for w in chrome))

    def test_photo_bar_is_not_touched_by_fullscreen(self):
        self.assertTrue(self.win.photo_bar.isHidden())      # fuera del modo foto sigue oculta al salir
        self.win.toggle_video_fullscreen()
        self.win.toggle_video_fullscreen()
        self.assertTrue(self.win.photo_bar.isHidden())

    def test_photo_bar_hides_in_fullscreen_and_corner_counter_takes_over(self):
        from pathlib import Path
        self.win.photo_paths = [Path("a.jpg"), Path("b.jpg")]     # simula modo foto sin cargar modelos
        self.win.photo_index = 1
        self.win.photo_bar.setVisible(True)
        self.win.toggle_video_fullscreen()
        self.assertTrue(self.win.photo_bar.isHidden())            # la barra con los botones se oculta...
        self.assertEqual(self.win.video._corner, "2 / 2")          # ...pero queda un contador discreto
        self.win.toggle_video_fullscreen()
        self.assertFalse(self.win.photo_bar.isHidden())           # al salir, vuelve a verse
        self.assertEqual(self.win.video._corner, "")


if __name__ == "__main__":
    unittest.main()
