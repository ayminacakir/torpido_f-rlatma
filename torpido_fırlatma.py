import subprocess
import sys
import cv2
import numpy as np
import math

class TorpidoFirlatma:
    def __init__(self):
        if sys.platform == 'win32':
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif sys.platform == 'darwin':
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_V4L)

        if not self.video_capture.isOpened():
            print("Kamera baslatilamadi.")
            exit()

        # Renk aralıklarını tanımlama
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])

        # Minimum piksel sayısı
        self.min_pixel = 500

        # Morfolojik işlemler için kernel
        self.kernel = np.ones((5, 5), np.uint8)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            return frame
        else:
            return None

    def find_largest_contour(self, contours):
        if contours:  # Kontur listesi boş değilse
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_pixel:
                return largest_contour
        return None  # Boş bir kontur listesi döndür

    def create_mask(self, hsv_frame, lower_color, upper_color):
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        return mask

    def apply_morphology(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def detect_elips(self, frame):
        subprocess.call("create_mask", shell=True)
        subprocess.call("apply_morpohology", shell=True)

        # Görüntüyü HSV renk uzayına dönüştür
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Maske oluştur
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red = cv2.inRange(hsv, self.lower_red, self.upper_red)

        # Morfolojik işlemler
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, self.kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, self.kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, self.kernel)

        # Kontur tespiti
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # En büyük konturları bul
        largest_contour_green = self.find_largest_contour(contours_green)
        largest_contour_yellow = self.find_largest_contour(contours_yellow)
        largest_contour_red = self.find_largest_contour(contours_red)

        # Algılanan halkaları işleme al
        detected_circles = []
        detected_angle = [None, None, None]
        if largest_contour_green is not None:
            ellipse = cv2.fitEllipse(largest_contour_green)
            center, axis_lengths, angle = ellipse
            center = (int(center[0]), int(center[1]))
            angle = int(angle)
            axis_length_major = int(axis_lengths[0] / 2)
            axis_length_minor = int(axis_lengths[1] / 2)
            aci = math.degrees(math.asin(axis_length_major / axis_length_minor))
            detected_angle[0] = aci
            detected_circles.append((center, axis_length_major, "Yeşil"))

        if largest_contour_yellow is not None:
            ellipse = cv2.fitEllipse(largest_contour_yellow)
            center, axis_lengths, angle = ellipse
            center = (int(center[0]), int(center[1]))
            angle = int(angle)
            axis_length_major = int(axis_lengths[0] / 2)
            axis_length_minor = int(axis_lengths[1] / 2)
            aci = math.degrees(math.asin(axis_length_major / axis_length_minor))
            detected_angle[1] = aci
            detected_circles.append((center, axis_length_major, "Sarı"))

        if largest_contour_red is not None:
            ellipse = cv2.fitEllipse(largest_contour_red)
            center, axis_lengths, angle = ellipse
            center = (int(center[0]), int(center[1]))
            angle = int(angle)
            axis_length_major = int(axis_lengths[0] / 2)
            axis_length_minor = int(axis_lengths[1] / 2)
            aci = math.degrees(math.asin(axis_length_major / axis_length_minor))
            detected_angle[2] = aci
            detected_circles.append((center, axis_length_major, "Kırmızı"))

        return detected_circles, detected_angle

    def draw_circles(self, frame, circles, angles):
        for (center, radius, color), angle in zip(circles, angles):
            if color == "Yeşil":
                cv2.ellipse(frame, center, (radius, radius), 0, 0, 360, (0, 255, 0), 2)
            elif color == "Sarı":
                cv2.ellipse(frame, center, (radius, radius), 0, 0, 360, (0, 255, 255), 2)
            elif color == "Kırmızı":
                cv2.ellipse(frame, center, (radius, radius), 0, 0, 360, (0, 0, 255), 2)

            cv2.putText(frame, f"{color} (Zorluk: {self.classify_difficulty(color)})",
                        (center[0] - radius, center[1] - radius - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if angle is not None:
                angle_text = f"Açı: {angle:.2f}°"
                cv2.putText(frame, angle_text, (center[0] - radius, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Merkezi işaretle
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Merkezi işaretle, kırmızı renkte
        return frame

    def classify_difficulty(self, color):
        if color == "Yeşil":
            return "Kolay"
        elif color == "Sarı":
            return "Orta"
        elif color == "Kırmızı":
            return "Zor"
        else:
            return "Bilinmeyen"

    def run(self):
        while True:
            frame = self.get_frame()
            circles, angles = self.detect_elips(frame)
            detected_circles = self.draw_circles(frame, circles, angles)

            cv2.imshow('Detected Circles', detected_circles)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                break

        self.release()

    def release(self):
        self.video_capture.release()


if __name__ == "__main__":
    torpido_firlatma = TorpidoFirlatma()
    torpido_firlatma.run()