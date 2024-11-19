import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
import sqlite3

# YOLO 모델 불러오기
model = YOLO('best.pt')  # 학습된 모델

class SmartPotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Pot")
        self.root.geometry("800x600")

        # Create database connection
        self.conn = sqlite3.connect('plant_growth.db')
        self.create_table()

        # UI Elements
        self.btn_select_image = tk.Button(root, text="이미지 선택", command=self.select_image)
        self.btn_select_image.pack(pady=10)

        self.btn_open_camera = tk.Button(root, text="카메라 열기", command=self.open_camera)
        self.btn_open_camera.pack(pady=10)

        self.text_cci = tk.Label(root, text="야호")
        self.text_cci.pack()

        self.label = tk.Label(root)
        self.label.pack()

        self.cap = None

    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
        
        self.show_frame()

    def show_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(cv2image)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.label.img_tk = img_tk  # Keep reference
                self.label.config(image=img_tk)

            # Continue updating the frame
            self.root.after(10, self.show_frame)

    ## Create WebCam Image Capture Function.

    def create_table(self):
        """식물 생장 데이터 저장 table"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS growth_data (
                id INTEGER PRIMARY KEY,
                width REAL,
                height REAL,
                cci REAL,
                result_class TEXT
            )
        ''')
        self.conn.commit()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # 이미지 로드 및 객체 인식 수행
            img = cv2.imread(file_path)
            results = model(img)
            annotated_frame = results[0].plot()  # 인식된 객체가 표시된 이미지 얻기
        
            # OpenCV 이미지를 PIL 이미지로 변환 후 Tkinter에서 사용
            img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img_pil)
            result_class = "Default"

            for r in results[0].boxes:
                result_class = model.names[int(r.cls)]

            # 이미지 레이블에 표시
            self.label.config(image=img_tk)
            self.label.image = img_tk  # 참조 유지

            width, height = self.calculate_plant_size(file_path, 100, 5, 200, 'green_mask.png', 28)
            cci = self.calculate_green_coverage(file_path)
            vegetation_index = self.calculate_vegetation_index(file_path)

            # 결과 출력 및 저장
            if width and height and result_class:
                print(f"식물의 너비: {width:.2f} cm, 높이: {height:.2f} cm, 피복비율: {cci:.2f}, 인식결과: {result_class}, 식생지수: {vegetation_index:.2f}")
                self.text_cci.config(text=f"식물의 너비: {width:.2f} cm, 높이: {height:.2f} cm, 피복비율: {cci:.2f}, 인식결과: {result_class}, 식생지수: {vegetation_index:.2f}")
                self.store_growth_data(width, height, cci, result_class)

                # 60cm 이상 [케이스 크기 고려하여 이 수치 변경 가능]
                if height > 60:
                    messagebox.showinfo("Notification", "The plant height is over 60 cm!")

                # 추천 알고리즘
                self.recommend_management(cci, vegetation_index)

            else:
                print("식물을 인식하지 못했습니다.")
                self.text_cci.config(text="식물을 인식하지 못했습니다.")

    def calculate_plant_size(self, image_path, known_distance, known_width, image_width_pixels, mask_output_path, focal_length_mm):
        # 이미지 읽기
        image = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 초록색 범위 정의 (HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # 초록색 마스크 생성
        green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
        
        # 마스크 이미지 저장
        cv2.imwrite(mask_output_path, green_mask)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 외접 직사각형 찾기
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 센서 크기 (예: 6.4mm x 4.8mm, 이는 일반적인 스마트폰 센서 크기 중 하나)
            sensor_width_mm = 6.4

            # 센서 크기와 이미지 해상도를 사용하여 픽셀 당 mm 계산
            pixel_size_mm = sensor_width_mm / image_width_pixels

            # 초점 거리, 센서 크기, 거리로 픽셀 당 cm 계산
            pixel_per_cm = (known_width * focal_length_mm) / (known_distance * pixel_size_mm)

            width_in_cm = w / pixel_per_cm
            height_in_cm = h / pixel_per_cm
            
            return width_in_cm, height_in_cm
        else:
            return None, None

    def calculate_green_coverage(self, image_path):
        # 이미지 읽기
        image = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 초록색 범위 정의 (HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # 초록색 마스크 생성
        green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
        
        # 초록색 픽셀 수 계산
        green_pixels = cv2.countNonZero(green_mask)
        
        # 전체 픽셀 수 계산
        total_pixels = image.shape[0] * image.shape[1]
        
        # 초록색 피복 비율 계산
        cci = green_pixels / total_pixels
        
        return cci

    def calculate_vegetation_index(self, image_path):
        # 이미지 읽기
        image = cv2.imread(image_path)
        # 소수점 제한 (오버플로우 방지)
        image_float = image.astype(np.float32)

        # RGB 채널 분리
        R = image_float[:, :, 2]
        G = image_float[:, :, 1]
        B = image_float[:, :, 0]

        # ExG (Excess Green) 계산
        exg = 2 * G - R - B

        # ExG의 평균 계산 (식생지수)
        vegetation_index = np.mean(exg)

        return vegetation_index

    def store_growth_data(self, width, height, cci, result_class):
        """식물 성장 데이터 저장"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO growth_data (width, height, cci, result_class)
            VALUES (?, ?, ?, ?)
        ''', (width, height, cci, result_class))
        self.conn.commit()

    def recommend_management(self, cci, vegetation_index):
        recommendation = "현재 상태를 유지해주세요"
        if cci > 0.5 or vegetation_index > 0.3:
            recommendation = "채광을 늘려주세요"
        elif cci < 0.2 or vegetation_index < 0.1:
            recommendation = "수분 및 영양 공급을 점검해주세요"

        print(f"작물 관리 추천: {recommendation}")
        messagebox.showinfo("Recommendation", recommendation)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartPotApp(root)
    root.mainloop()