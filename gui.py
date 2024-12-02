import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# plt font error solve
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# YOLO 모델 불러오기
initial_model = YOLO('best2.pt')  # 해충 / 비해충 구분용 Model
detailed_model = YOLO('best.pt')  # 해충 세부 종 구분 Model

plant_model = YOLO('best4.pt')  # 식물 질병 구분 Model

class SmartPotUI:
    def __init__(self, root, logic):
        self.logic = logic
        self.root = root
        self.root.title("Smart Pot")
        self.root.geometry("1200x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # UI Elements
        self.frame_left = ctk.CTkFrame(root, width=200, corner_radius=15)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_select_image = ctk.CTkButton(self.frame_left, text="이미지 선택", command=self.logic.select_image, font=('Nanum Gothic', 12))
        self.btn_select_image.pack(pady=10)

        self.btn_open_camera = ctk.CTkButton(self.frame_left, text="카메라 열기", command=self.logic.open_camera, font=('Nanum Gothic', 12))
        self.btn_open_camera.pack(pady=10)

        self.btn_predict_water_cycle = ctk.CTkButton(self.frame_left, text="수분 공급 주기 예측", command=self.logic.predict_water_cycle, font=('Nanum Gothic', 12))
        self.btn_predict_water_cycle.pack(pady=10)

        self.text_cci = ctk.CTkLabel(self.frame_left, text="이미지를 불러와주세요", wraplength=150, font=('Nanum Gothic', 12))
        self.text_cci.pack(pady=20)

        self.btn_show_chart = ctk.CTkButton(self.frame_left, text="데이터 시각화", command=self.logic.visualize_data, font=('Nanum Gothic', 12))
        self.btn_show_chart.pack(pady=10)

        self.frame_right = ctk.CTkFrame(root, corner_radius=15)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.label = ctk.CTkLabel(self.frame_right)
        self.label.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.get_tk_widget().pack()

        self.cap = None
        self.max_display_width = 800
        self.max_display_height = 600

    def update_camera_frame(self, img_tk):
        self.label.img_tk = img_tk  # Keep reference
        self.label.configure(image=img_tk)

    def update_text(self, text):
        self.text_cci.configure(text=text)

class SmartPotLogic:
    def __init__(self, ui):
        self.ui = ui
        self.conn = sqlite3.connect('plant_growth.db')
        self.add_disease_column_if_not_exists()
        self.create_table()
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
                img_tk = ImageTk.PhotoImage(image=self.resize_image(img_pil))
                self.ui.update_camera_frame(img_tk)

            # Continue updating the frame
            self.ui.root.after(10, self.show_frame)

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS growth_data (
                id INTEGER PRIMARY KEY,
                width REAL,
                height REAL,
                cci REAL,
                result_class TEXT,
                disease TEXT
            )
        ''')
        self.conn.commit()

    def add_disease_column_if_not_exists(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("ALTER TABLE growth_data ADD COLUMN disease TEXT")
        except sqlite3.OperationalError:
            # 컬럼이 이미 존재하는 경우 예외 발생, 무시함
            pass
        self.conn.commit()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)

            # Initial detection for pests
            results = initial_model(img)
            annotated_frame = results[0].plot()

            img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(self.resize_image(img_pil))
            pest_detected = False
            species = "Unknown"

            for r in results[0].boxes:
                object_class = initial_model.names[int(r.cls)]
                if object_class is not None:
                    pest_detected = True
                    species_results = detailed_model(img)
                    for sr in species_results[0].boxes:
                        species = detailed_model.names[int(sr.cls)]
                    break

            # Plant disease detection
            disease_detected = False
            disease_name = "None"
            disease_results = plant_model(img)
            for dr in disease_results[0].boxes:
                disease_name = plant_model.names[int(dr.cls)]
                if (len(disease_name) > 15):
                    disease_detected = True
                    break

            # Display the image in the label
            self.ui.update_camera_frame(img_tk)

            width, height = self.calculate_plant_size(file_path, 100, 5, 200, 'green_mask.png', 28)
            cci = self.calculate_green_coverage(file_path)
            vegetation_index = self.calculate_vegetation_index(file_path)

            if width and height:
                result_text = f"식물의 너비: {width:.2f} cm, 높이: {height:.2f} cm, 피복비율: {cci:.2f}, 식생지수: {vegetation_index:.2f}"
                print(result_text)
                self.ui.update_text(result_text)
                self.store_growth_data(width, height, cci, species, disease_name)

                if pest_detected:
                    messagebox.showinfo("Detection Result", f"해충 종: {species}")
                else:
                    messagebox.showinfo("Detection Result", "해충 인식되지 않음")

                if disease_detected:
                    messagebox.showinfo("Disease Detection", f"식물 질병: {disease_name}")
                else:
                    messagebox.showinfo("Disease Detection", "질병 인식되지 않음")

                if height > 60:
                    messagebox.showinfo("Notification", "식물이 화분 보다 클 수 있습니다")

                self.recommend_management(cci, vegetation_index)

            else:
                print("식물을 인식하지 못했습니다.")
                self.ui.update_text("식물을 인식하지 못했습니다.")

    def resize_image(self, img):
        width, height = img.size
        if width > self.ui.max_display_width or height > self.ui.max_display_height:
            ratio = min(self.ui.max_display_width / width, self.ui.max_display_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            return img.resize(new_size, Image.LANCZOS)
        return img
    
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

    def store_growth_data(self, width, height, cci, result_class, disease):
        """
        식물 성장 데이터 저장
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO growth_data (width, height, cci, result_class, disease)
            VALUES (?, ?, ?, ?, ?)
        ''', (width, height, cci, result_class, disease))
        self.conn.commit()

    def recommend_management(self, cci, vegetation_index):
        recommendation = "현재 상태를 유지해주세요"
        if cci > 0.5 or vegetation_index > 0.3:
            recommendation = "채광을 늘려주세요"
        elif cci < 0.2 or vegetation_index < 0.1:
            recommendation = "수분 및 영양 공급을 점검해주세요"

        print(f"작물 관리 추천: {recommendation}")
        messagebox.showinfo("Recommendation", recommendation)

    def predict_water_cycle(self):
        np.random.seed(0)
        days = np.arange(1, 31)
        soil_moisture = np.random.uniform(low=0, high=100, size=30)  # Random soil moisture values

        water_need_days = days[soil_moisture < 30]
        X = water_need_days.reshape(-1, 1)
        y = np.roll(water_need_days, -1)[:-1] - water_need_days[:-1]

        model = LinearRegression()
        model.fit(X[:-1], y)
        predicted_cycle = model.predict(X)

        self.ui.ax.clear()
        self.ui.ax.scatter(days, soil_moisture, label='토양 습도')
        self.ui.ax.plot(X, model.predict(X), color='red', label='예측된 수분 공급 주기')
        self.ui.ax.axhline(y=30, color='green', linestyle='--', label='수분 공급 한계점')
        self.ui.ax.set_xlabel('일')
        self.ui.ax.set_ylabel('토양 습도')
        self.ui.ax.set_title('토양 습도 수준에 따른 수분 공급 주기 예측')
        self.ui.ax.legend()
        self.ui.canvas.draw()

        print("수분 공급 주기 예측:", predicted_cycle)

    def visualize_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT width, height, cci FROM growth_data")
        data = cursor.fetchall()
        if data:
            widths, heights, ccis = zip(*data)
            self.ui.ax.clear()
            self.ui.ax.plot(widths, label='너비 (cm)', color='blue')
            self.ui.ax.plot(heights, label='높이 (cm)', color='red')
            self.ui.ax.plot(ccis, label='피복비율', color='green')
            self.ui.ax.set_title('식물 성장 데이터 시각화')
            self.ui.ax.set_xlabel('데이터 포인트')
            self.ui.ax.set_ylabel('값')
            self.ui.ax.legend()
            self.ui.canvas.draw()
        else:
            messagebox.showinfo("데이터 없음", "저장된 성장 데이터가 없습니다.")

if __name__ == "__main__":
    root = ctk.CTk()
    logic = SmartPotLogic(None)
    app = SmartPotUI(root, logic)
    logic.ui = app
    root.mainloop()
