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
import threading
import time
import random
import tkinter as tk  # Import standard tkinter
import logging
import datetime

# Configure logging
logging.basicConfig(
    filename='smart_farm.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# plt font error solve
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# YOLO 모델 불러오기
initial_model = YOLO('insect_conf_model.pt')  # 해충 / 비해충 구분용 Model
detailed_model = YOLO('insect_spec_model.pt')  # 해충 세부 종 구분 Model
plant_model = YOLO('plant_doc_model.pt')  # 식물 질병 구분 Model

# Flag to use Arduino or dummy data
USE_ARDUINO = False  # Set to True when Arduino is available

class ScrollableFrame(ctk.CTkFrame):
    """
    A scrollable frame that can be used to contain other widgets.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create a canvas and a vertical scrollbar for scrolling it
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(self.canvas, fg_color="black")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to scroll
        self.scrollable_frame.bind("<Enter>", self._bind_to_mousewheel)
        self.scrollable_frame.bind("<Leave>", self._unbind_from_mousewheel)

    def _bind_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        # For Windows
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # For macOS, you might need to adjust the scroll direction
        # self.canvas.yview_scroll(int(-1*(event.delta)), "units")

class SmartFarmUI:
    def __init__(self, root, logic):
        self.logic = logic
        self.root = root
        self.root.title("Smart Pot Dashboard")
        self.root.geometry("1800x1000")  # Increased size for better layout
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # 좌측 패널
        self.frame_left = ctk.CTkFrame(root, width=300, corner_radius=15)
        self.frame_left.pack(side="left", fill="y", padx=10, pady=10)

        # 이미지 및 카메라 관련 버튼
        self.btn_select_image = ctk.CTkButton(
            self.frame_left, text="이미지 선택", command=self.logic.select_image, font=('Nanum Gothic', 14))
        self.btn_select_image.pack(pady=15)

        self.btn_open_camera = ctk.CTkButton(
            self.frame_left, text="카메라 열기", command=self.logic.open_camera, font=('Nanum Gothic', 14))
        self.btn_open_camera.pack(pady=15)

        # 사진 촬영 버튼 추가
        self.btn_capture_image = ctk.CTkButton(
            self.frame_left, text="사진 촬영", command=self.logic.capture_image, font=('Nanum Gothic', 14))
        self.btn_capture_image.pack(pady=15)

        self.btn_predict_water_cycle = ctk.CTkButton(
            self.frame_left, text="수분 공급 주기 예측", command=self.logic.predict_water_cycle, font=('Nanum Gothic', 14))
        self.btn_predict_water_cycle.pack(pady=15)

        self.btn_show_chart = ctk.CTkButton(
            self.frame_left, text="데이터 시각화", command=self.logic.visualize_data, font=('Nanum Gothic', 14))
        self.btn_show_chart.pack(pady=15)

        # 환경 모니터링 정보
        self.label_env_title = ctk.CTkLabel(
            self.frame_left, text="환경 모니터링", font=('Nanum Gothic', 16))
        self.label_env_title.pack(pady=(30, 15))

        self.label_light = ctk.CTkLabel(
            self.frame_left, text="조도: N/A", font=('Nanum Gothic', 14))
        self.label_light.pack(pady=8)

        self.label_co2 = ctk.CTkLabel(
            self.frame_left, text="CO₂ 농도: N/A", font=('Nanum Gothic', 14))
        self.label_co2.pack(pady=8)

        self.label_ph = ctk.CTkLabel(
            self.frame_left, text="pH 농도: N/A", font=('Nanum Gothic', 14))
        self.label_ph.pack(pady=8)

        self.label_water_level = ctk.CTkLabel(
            self.frame_left, text="물탱크 잔량: N/A", font=('Nanum Gothic', 14))
        self.label_water_level.pack(pady=8)

        # **New Labels for Temperature and Humidity**
        self.label_temperature = ctk.CTkLabel(
            self.frame_left, text="온도: N/A", font=('Nanum Gothic', 14))
        self.label_temperature.pack(pady=8)

        self.label_humidity = ctk.CTkLabel(
            self.frame_left, text="습도: N/A", font=('Nanum Gothic', 14))
        self.label_humidity.pack(pady=8)

        # 하드웨어 제어
        self.label_control_title = ctk.CTkLabel(
            self.frame_left, text="하드웨어 제어", font=('Nanum Gothic', 16))
        self.label_control_title.pack(pady=(30, 15))

        self.btn_toggle_pump = ctk.CTkButton(
            self.frame_left, text="급수 펌프 토글", command=self.logic.toggle_pump, font=('Nanum Gothic', 14))
        self.btn_toggle_pump.pack(pady=10)

        self.btn_toggle_led = ctk.CTkButton(
            self.frame_left, text="LED 조명 토글", command=self.logic.toggle_led, font=('Nanum Gothic', 14))
        self.btn_toggle_led.pack(pady=10)

        # 상태 메시지
        self.label_status = ctk.CTkLabel(
            self.frame_left, text="상태: 초기화 중", font=('Nanum Gothic', 14))
        self.label_status.pack(pady=(50, 20))

        # 우측 패널 with Scrollable Frame
        self.frame_right = ctk.CTkFrame(root, corner_radius=15)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.scrollable_frame = ScrollableFrame(self.frame_right)
        self.scrollable_frame.pack(fill="both", expand=True)

        # **Add "Refresh All Data" Button at the Top Right**
        self.btn_refresh_all = ctk.CTkButton(
            self.scrollable_frame.scrollable_frame,
            text="데이터 최신화",
            command=self.logic.visualize_and_predict,
            font=('Nanum Gothic', 14)
        )
        self.btn_refresh_all.pack(pady=10)

        # 상단: 카메라 및 이미지 표시
        self.frame_image = ctk.CTkFrame(self.scrollable_frame.scrollable_frame, corner_radius=10, fg_color="black")
        self.frame_image.pack(fill="x", pady=10)

        self.label_image = ctk.CTkLabel(self.frame_image, text="", fg_color="black")
        self.label_image.pack(pady=10)

        # 중단: 이미지 분석 결과 표시
        self.frame_results = ctk.CTkFrame(self.scrollable_frame.scrollable_frame, corner_radius=10, fg_color="black")
        self.frame_results.pack(fill="x", pady=10)

        self.label_pest = ctk.CTkLabel(
            self.frame_results, text="해충: N/A", font=('Nanum Gothic', 14), text_color="white", fg_color="black")
        self.label_pest.pack(pady=5)

        self.label_disease = ctk.CTkLabel(
            self.frame_results, text="질병: N/A", font=('Nanum Gothic', 14), text_color="white", fg_color="black")
        self.label_disease.pack(pady=5)

        self.label_size = ctk.CTkLabel(
            self.frame_results, text="식물 크기: N/A", font=('Nanum Gothic', 14), text_color="white", fg_color="black")
        self.label_size.pack(pady=5)

        # 하단: 데이터 시각화 그래프
        self.frame_charts = ctk.CTkFrame(self.scrollable_frame.scrollable_frame, corner_radius=10, fg_color="black")
        self.frame_charts.pack(fill="both", expand=True, pady=10)

        # Initialize Matplotlib Figure with black background
        self.fig, self.gs = plt.subplots(figsize=(12, 8), constrained_layout=True)
        self.fig.patch.set_facecolor('black')  # Figure background
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_charts)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_camera_frame(self, img_tk):
        self.label_image.img_tk = img_tk  # Keep reference
        self.label_image.configure(image=img_tk)

    def update_environment_info(self, light, co2, ph, water_level, temperature=None, humidity=None):
        self.label_light.configure(text=f"조도: {light} lux")
        self.label_co2.configure(text=f"CO₂ 농도: {co2} ppm")
        self.label_ph.configure(text=f"pH 농도: {ph}")
        self.label_water_level.configure(text=f"물탱크 잔량: {water_level}%")
        
        # **Update Temperature and Humidity if Provided**
        if temperature is not None:
            self.label_temperature.configure(text=f"온도: {temperature}°C")
        if humidity is not None:
            self.label_humidity.configure(text=f"습도: {humidity}%")

    def update_status(self, status):
        self.label_status.configure(text=f"상태: {status}")

    def update_analysis_results(self, pest, disease, size_text):
        self.label_pest.configure(text=f"해충: {pest}")
        self.label_disease.configure(text=f"질병: {disease}")
        self.label_size.configure(text=f"식물 크기: {size_text}")

class SmartFarmLogic:
    def __init__(self, ui=None):
        self.ui = ui
        self.conn = sqlite3.connect('plant_growth.db', check_same_thread=False)  # Allow access from multiple threads
        self.cap = None
        self.current_frame = None  # 현재 프레임 저장용 변수
        self.arduino = None
        self.dummy_data_thread = None
        self.reading_thread = None
        self.create_table()
        self.is_predicting = False  # Flag to prevent multiple predictions at the same time

    def ensure_columns_exist(self, cursor, table_name, required_columns):
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [info[1] for info in cursor.fetchall()]
        for column in required_columns:
            if column not in existing_columns:
                if column == 'disease':
                    cursor.execute("ALTER TABLE growth_data ADD COLUMN disease TEXT")
                elif column == 'vegetation_index':
                    cursor.execute("ALTER TABLE growth_data ADD COLUMN vegetation_index REAL")
        self.conn.commit()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS growth_data (
                id INTEGER PRIMARY KEY,
                width REAL,
                height REAL,
                cci REAL,
                vegetation_index REAL,
                result_class TEXT,
                disease TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.ensure_columns_exist(cursor, 'growth_data', ['vegetation_index', 'disease'])
        self.conn.commit()

    def set_ui(self, ui):
        self.ui = ui
        if USE_ARDUINO:
            # Initialize Arduino communication
            try:
                import serial
                self.arduino = serial.Serial('COM3', 9600, timeout=1)  # Update COM port as needed
                time.sleep(2)  # Wait for Arduino to reset
                self.ui.update_status("Arduino 연결됨")
                self.reading_thread = threading.Thread(target=self.read_from_arduino, daemon=True)
                self.reading_thread.start()
            except (ImportError, serial.SerialException):
                self.ui.update_status("Arduino 연결 실패")
                self.arduino = None
        else:
            # Use dummy data
            self.ui.update_status("더미 데이터 사용 중")
            self.dummy_data_thread = threading.Thread(target=self.generate_dummy_data, daemon=True)
            self.dummy_data_thread.start()

        # Start automatic visualization and prediction
        self.schedule_visualize_and_predict()

    def schedule_visualize_and_predict(self):
        """
        Schedule the visualize_and_predict method to run every 60 seconds.
        """
        self.visualize_and_predict()
        # Schedule the next call in 60,000 milliseconds (60 seconds)
        self.ui.root.after(60000, self.schedule_visualize_and_predict)

    def visualize_and_predict(self):
        """
        Perform data visualization and, if sufficient data is available, perform water cycle prediction.
        """
        try:
            self.visualize_data()
            # Check if enough data is accumulated for prediction (e.g., 30 data points)
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM growth_data WHERE timestamp >= datetime('now', '-30 minutes')")
            count = cursor.fetchone()[0]
            logging.debug(f"Data points in the last 30 minutes: {count}")
            if count >= 30 and not self.is_predicting:
                self.is_predicting = True
                threading.Thread(target=self.perform_prediction, daemon=True).start()
        except Exception as e:
            logging.error(f"Error in visualize_and_predict: {e}")

    def perform_prediction(self):
        """
        Perform water cycle prediction and act upon the result.
        """
        try:
            self.predict_water_cycle()
            # After prediction, reset the flag
        except Exception as e:
            logging.error(f"Error in perform_prediction: {e}")
        finally:
            self.is_predicting = False

    def generate_dummy_data(self):
        while True:
            # Generate random dummy data
            light = random.randint(100, 1000)  # lux
            co2 = random.randint(300, 800)     # ppm
            ph = round(random.uniform(5.5, 7.5), 2)
            water_level = random.randint(0, 100)  # percentage
            temperature = round(random.uniform(15.0, 30.0), 2)  # °C
            humidity = random.randint(30, 90)  # %

            # Update the UI with dummy data if ui is set
            if self.ui:
                self.ui.update_environment_info(
                    light=light,
                    co2=co2,
                    ph=ph,
                    water_level=water_level,
                    temperature=temperature,
                    humidity=humidity
                )
            time.sleep(5)  # Update every 5 seconds

    def read_from_arduino(self):
        while True:
            if self.arduino and self.arduino.in_waiting > 0:
                try:
                    line = self.arduino.readline().decode().strip()
                    if line:
                        # Expected data format: "조도,수위,토양습도,이산화탄소,PH,온도,습도"
                        data = line.split(",")
                        if len(data) == 7:
                            light = float(data[0])  # 조도
                            water_level = float(data[1])  # 수위
                            soil_humidity = float(data[2])  # 토양습도
                            co2 = float(data[3])  # 이산화탄소
                            ph = float(data[4])  # PH
                            temperature = float(data[5])  # 온도
                            humidity = float(data[6])  # 습도

                            if self.ui:
                                self.ui.update_environment_info(
                                    light=light,
                                    co2=co2,
                                    ph=ph,
                                    water_level=water_level,
                                    temperature=temperature,
                                    humidity=humidity
                                )
                            # Store the data in the database
                            self.store_growth_data_from_sensors(soil_humidity)
                        else:
                            logging.warning(f"Unexpected data format: {line}")
                except Exception as e:
                    logging.error(f"데이터 파싱 오류: {e}")
            time.sleep(1)  # Data reading interval

    def open_camera(self):
        if self.cap is None:
            # Attempt to open the default camera (index 0)
            self.cap = cv2.VideoCapture(1)  # 웹캠 인덱스 1
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)  # Fallback to index 0
                if not self.cap.isOpened():
                    messagebox.showerror("카메라 오류", "웹캠을 열 수 없습니다.")
                    logging.error("Failed to open the camera.")
                    return
        self.show_frame()

    def show_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()  # 현재 프레임 저장
                # Convert the frame to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(cv2image)
                img_tk = ImageTk.PhotoImage(image=self.resize_image(img_pil))
                self.ui.update_camera_frame(img_tk)
        # Continue updating the frame
        self.ui.root.after(10, self.show_frame)

    def select_image(self):
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("파일 오류", "이미지를 불러오는 데 실패했습니다.")
                    logging.error(f"Failed to load image: {file_path}")
                    return

                analysis_results = self.analyze_image(img)
                if analysis_results is None:
                    messagebox.showerror("데이터 오류", "식물 크기 계산에 실패했습니다.")
                    self.ui.update_analysis_results("인식되지 않음", "인식되지 않음", "인식되지 않음")
                    logging.error("Failed to analyze image.")
                    return

                width, height, cci, vegetation_index, species, disease_name = analysis_results

                annotated_frame = initial_model(img)[0].plot()
                img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(self.resize_image(img_pil))

                # Define size_text before passing
                size_text = f"너비: {width:.2f} cm, 높이: {height:.2f} cm"

                # Update UI
                self.ui.update_camera_frame(img_tk)
                self.ui.update_analysis_results(
                    species if species != "Unknown" else "인식되지 않음",
                    disease_name if disease_name != "None" else "인식되지 않음",
                    size_text
                )
                logging.debug(f"Updated analysis results: pest={species}, disease={disease_name}, size_text={size_text}")

                self.store_growth_data(width, height, cci, vegetation_index, species, disease_name)

                if species != "Unknown":
                    messagebox.showinfo("Detection Result", f"해충 종: {species}")
                else:
                    messagebox.showinfo("Detection Result", "해충 인식되지 않음")

                if disease_name != "None":
                    messagebox.showinfo("Disease Detection", f"식물 질병: {disease_name}")
                else:
                    messagebox.showinfo("Disease Detection", "질병 인식되지 않음")

                if height > 60:
                    messagebox.showinfo("Notification", "식물이 화분 보다 클 수 있습니다")

                self.recommend_management(cci, vegetation_index)
            else:
                print("식물을 인식하지 못했습니다.")
                self.ui.update_analysis_results("인식되지 않음", "인식되지 않음", "인식되지 않음")
                logging.info("No image selected or failed to recognize plant.")
        except Exception as e:
            messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {e}")
            logging.error(f"Error in select_image: {e}")

    def capture_image(self):
        if self.cap is not None and self.cap.isOpened():
            if self.current_frame is None:
                messagebox.showerror("캡처 오류", "현재 프레임이 없습니다.")
                logging.error("No current frame available for capture.")
                return

            try:
                frame = self.current_frame.copy()
                img = frame

                analysis_results = self.analyze_image(img)
                if analysis_results is None:
                    messagebox.showerror("데이터 오류", "식물 크기 계산에 실패했습니다.")
                    self.ui.update_analysis_results("인식되지 않음", "인식되지 않음", "인식되지 않음")
                    logging.error("Failed to analyze captured image.")
                    return

                width, height, cci, vegetation_index, species, disease_name = analysis_results

                annotated_frame = initial_model(img)[0].plot()
                img_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(self.resize_image(img_pil))

                # Define size_text before passing
                size_text = f"너비: {width:.2f} cm, 높이: {height:.2f} cm"

                # Update UI
                self.ui.update_camera_frame(img_tk)
                self.ui.update_analysis_results(
                    species if species != "Unknown" else "인식되지 않음",
                    disease_name if disease_name != "None" else "인식되지 않음",
                    size_text
                )
                logging.debug(f"Updated analysis results: pest={species}, disease={disease_name}, size_text={size_text}")

                self.store_growth_data(width, height, cci, vegetation_index, species, disease_name)

                if species != "Unknown":
                    messagebox.showinfo("Detection Result", f"해충 종: {species}")
                else:
                    messagebox.showinfo("Detection Result", "해충 인식되지 않음")

                if disease_name != "None":
                    messagebox.showinfo("Disease Detection", f"식물 질병: {disease_name}")
                else:
                    messagebox.showinfo("Disease Detection", "질병 인식되지 않음")

                if height > 60:
                    messagebox.showinfo("Notification", "식물이 화분 보다 클 수 있습니다")

                self.recommend_management(cci, vegetation_index)
            except Exception as e:
                messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {e}")
                logging.error(f"Error in capture_image: {e}")
        else:
            messagebox.showerror("카메라 오류", "카메라가 열려 있지 않습니다.")

    def analyze_image(self, img):
        try:
            # Initial detection for pests
            results = initial_model(img)
            annotated_frame = results[0].plot()

            pest_detected = False
            species = "Unknown"

            for r in results[0].boxes:
                object_class = initial_model.names[int(r.cls)]
                if object_class is not None:
                    if r.conf >= 0.55:  # confidence score 0.55 이상인 경우
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
                if len(disease_name) > 15:
                    disease_detected = True
                    break

            # 크기 및 식생지수 계산
            width, height = self.calculate_plant_size_from_image(img, 100, 5, 200, 'green_mask.png', 28)
            cci = self.calculate_green_coverage_from_image(img)
            vegetation_index = self.calculate_vegetation_index_from_image(img)

            if width is None or height is None or cci is None or vegetation_index is None:
                return None

            return width, height, cci, vegetation_index, species, disease_name

        except Exception as e:
            logging.error(f"Image analysis error: {e}")
            return None

    def resize_image(self, img):
        width, height = img.size
        max_width = 800
        max_height = 600
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            return img.resize(new_size, Image.LANCZOS)
        return img

    def calculate_plant_size_from_image(self, img, known_distance, known_width, image_width_pixels, mask_output_path, focal_length_mm):
        try:
            image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 초록색 범위 정의 (HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])

            # 초록색 마스크 생성
            green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

            # 마스크 이미지 저장 (필요 시)
            cv2.imwrite(mask_output_path, green_mask)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 가장 큰 윤곽선 선택
                largest_contour = max(contours, key=cv2.contourArea)

                # 외접 직사각형 찾기
                x, y, w, h = cv2.boundingRect(largest_contour)

                # 센서 크기 (예: 6.4mm x 4.8mm)
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
        except Exception as e:
            logging.error(f"calculate_plant_size_from_image error: {e}")
            return None, None

    def calculate_green_coverage_from_image(self, img):
        try:
            image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 초록색 범위 정의 (HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])

            # 초록색 마스크 생성
            green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

            # 초록색 픽셀 수 계산
            green_pixels = cv2.countNonZero(green_mask)

            # 전체 픽셀 수 계산
            total_pixels = img.shape[0] * img.shape[1]

            # 초록색 피복 비율 계산
            cci = green_pixels / total_pixels

            return cci
        except Exception as e:
            logging.error(f"calculate_green_coverage_from_image error: {e}")
            return None

    def calculate_vegetation_index_from_image(self, img):
        try:
            # 소수점 제한 (오버플로우 방지)
            image_float = img.astype(np.float32)

            # RGB 채널 분리
            R = image_float[:, :, 2]
            G = image_float[:, :, 1]
            B = image_float[:, :, 0]

            # ExG (Excess Green) 계산
            exg = 2 * G - R - B

            # ExG의 평균 계산 (식생지수)
            vegetation_index = np.mean(exg)

            return vegetation_index
        except Exception as e:
            logging.error(f"calculate_vegetation_index_from_image error: {e}")
            return None

    def store_growth_data(self, width, height, cci, vegetation_index, result_class, disease):
        """
        식물 성장 데이터 저장
        """
        try:
            width = float(width)
            height = float(height)
            cci = float(cci)
            vegetation_index = float(vegetation_index)
        except ValueError as e:
            logging.error(f"Data conversion error: {e}")
            messagebox.showerror("데이터 오류", f"잘못된 데이터 형식이 있습니다: {e}")
            return

        # Ensure result_class and disease are strings, not bytes
        if isinstance(result_class, bytes):
            result_class = result_class.decode('utf-8', errors='ignore')
        if isinstance(disease, bytes):
            disease = disease.decode('utf-8', errors='ignore')

        logging.debug(f"Inserting data: width={width}, height={height}, cci={cci}, vegetation_index={vegetation_index}, result_class={result_class}, disease={disease}")

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO growth_data (width, height, cci, vegetation_index, result_class, disease)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (width, height, cci, vegetation_index, result_class, disease))
        self.conn.commit()

    def store_growth_data_from_sensors(self, cci):
        """
        Store soil moisture (cci) data from sensors.
        This method is called when data is received from Arduino or dummy data.
        """
        try:
            cci = float(cci)
        except ValueError as e:
            logging.error(f"CCI data conversion error: {e}")
            return

        logging.debug(f"Inserting sensor data: cci={cci}")

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO growth_data (cci)
            VALUES (?)
        ''', (cci,))
        self.conn.commit()

    def recommend_management(self, cci, vegetation_index):
        recommendation = "현재 상태를 유지해주세요"
        if cci > 0.5 or vegetation_index > 0.3:
            recommendation = "채광을 늘려주세요"
        elif cci < 0.2 or vegetation_index < 0.1:
            recommendation = "수분 및 영양 공급을 점검해주세요"

        print(f"작물 관리 추천: {recommendation}")
        self.ui.update_status(f"작물 관리 추천: {recommendation}")
        messagebox.showinfo("Recommendation", recommendation)

    def predict_water_cycle(self):
        try:
            cursor = self.conn.cursor()
            # Retrieve soil moisture data from the last 30 minutes
            cursor.execute("""
                SELECT timestamp, cci FROM growth_data 
                WHERE timestamp >= datetime('now', '-30 minutes') 
                ORDER BY timestamp ASC
            """)
            data = cursor.fetchall()
            if len(data) < 2:
                logging.info("Not enough data for prediction.")
                return

            # Extract time (in minutes) and soil moisture (cci)
            times = []
            soil_moisture = []
            for row in data:
                timestamp, cci = row
                # Calculate minutes since the first timestamp
                first_time = data[0][0]
                current_time = row[0]
                time_diff = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(first_time, '%Y-%m-%d %H:%M:%S')
                minutes = time_diff.total_seconds() / 60
                times.append(minutes)
                soil_moisture.append(cci)

            # Convert to numpy arrays
            X = np.array(times).reshape(-1, 1)
            y = np.array(soil_moisture)

            # Fit Linear Regression model
            model = LinearRegression()
            model.fit(X, y)
            predicted = model.predict(X)

            # Predict the next minute's soil moisture
            next_minute = np.array([[X[-1][0] + 1]])
            predicted_next = model.predict(next_minute)[0]

            logging.debug(f"Predicted next soil moisture (cci): {predicted_next}")

            # Update the visualization with prediction
            self.ui.fig.clf()  # Clear the figure

            # Create a GridSpec layout with 2 rows and 2 columns
            gs = self.ui.fig.add_gridspec(2, 2)

            # 토양 습도 라인 차트
            ax1 = self.ui.fig.add_subplot(gs[0, 0], facecolor='black')
            ax1.plot(times, soil_moisture, label='토양 습도', color='cyan')
            ax1.plot(times, predicted, label='예측 토양 습도', color='orange', linestyle='--')
            ax1.set_xlabel('시간 (분)', color='white')
            ax1.set_ylabel('토양 습도 (CCI)', color='white')
            ax1.set_title('토양 습도 변화 추이', color='white')
            ax1.legend()

            # 수분 공급 주기 예측 결과
            ax2 = self.ui.fig.add_subplot(gs[0, 1], facecolor='black')
            ax2.text(0.5, 0.5, f"예측된 다음 수분 공급 CCI: {predicted_next:.2f}", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='white', transform=ax2.transAxes)
            ax2.axis('off')  # Hide the axes

            # 토양 습도 히스토그램
            ax3 = self.ui.fig.add_subplot(gs[1, 0], facecolor='black')
            ax3.hist(soil_moisture, bins=10, color='purple', alpha=0.7)
            ax3.set_xlabel('토양 습도 (CCI)', color='white')
            ax3.set_ylabel('빈도수', color='white')
            ax3.set_title('토양 습도 분포', color='white')

            # 식생지수 바 차트
            ax4 = self.ui.fig.add_subplot(gs[1, 1], facecolor='black')
            avg_vegetation = self.calculate_vegetation_index_from_cycle(soil_moisture)
            ax4.bar(['식생지수'], [avg_vegetation], color='green')
            ax4.set_ylabel('식생지수', color='white')
            ax4.set_title('평균 식생지수', color='white')

            self.ui.fig.tight_layout()
            self.ui.canvas.draw()

            # Display the prediction result in the UI
            self.ui.update_status(f"수분 공급 주기 예측: CCI={predicted_next:.2f}")

            # If predicted soil moisture is below threshold, activate pump
            if predicted_next < 0.3:  # Example threshold
                self.toggle_pump()
                messagebox.showinfo("수분 공급", "수분 공급 주기가 도래했습니다. 급수 펌프를 작동시켰습니다.")
                logging.info("Pump activated based on water cycle prediction.")

        except Exception as e:
            messagebox.showerror("예측 오류", f"수분 공급 주기 예측 중 오류가 발생했습니다: {e}")
            logging.error(f"Error in predict_water_cycle: {e}")

    def calculate_vegetation_index_from_cycle(self, soil_moisture):
        # 간단한 식생지수 계산 예시 (실제 로직에 맞게 수정 필요)
        return np.mean(soil_moisture) / 100

    def visualize_data(self):
        """
        Visualize the collected growth data.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT width, height, cci, vegetation_index FROM growth_data")
            data = cursor.fetchall()
            if data:
                widths = []
                heights = []
                ccis = []
                vegetation_indices = []
                invalid_rows = []
                for row in data:
                    try:
                        w = float(row[0])
                        h = float(row[1])
                        c = float(row[2])
                        v = float(row[3])
                        widths.append(w)
                        heights.append(h)
                        ccis.append(c)
                        vegetation_indices.append(v)
                    except Exception as e:
                        invalid_rows.append(row)
                        logging.warning(f"Skipping invalid row: {row}, error: {e}")

                if invalid_rows:
                    logging.info(f"Found and skipped {len(invalid_rows)} invalid rows.")

                if not widths:
                    messagebox.showinfo("데이터 없음", "모든 데이터가 유효하지 않습니다.")
                    return

                self.ui.fig.clf()  # Clear the entire figure

                # Create a GridSpec layout with 2 rows and 2 columns
                gs = self.ui.fig.add_gridspec(2, 2)

                try:
                    # 피복 비율 파이 차트
                    ax1 = self.ui.fig.add_subplot(gs[0, 0], facecolor='black')
                    labels = ['피복 비율', '나머지']
                    sizes = [np.mean(ccis), 1 - np.mean(ccis)]
                    colors = ['green', 'lightgrey']
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    ax1.set_title('평균 피복 비율', color='white')

                    # 식생지수 라인 차트
                    ax2 = self.ui.fig.add_subplot(gs[0, 1], facecolor='black')
                    ax2.plot(vegetation_indices, label='식생지수', color='yellow')
                    ax2.set_xlabel('데이터 포인트', color='white')
                    ax2.set_ylabel('식생지수', color='white')
                    ax2.set_title('식생지수 변화 추이', color='white')
                    ax2.legend()

                    # 식물 너비 및 높이 바 차트
                    ax3 = self.ui.fig.add_subplot(gs[1, 0], facecolor='black')
                    indices = np.arange(len(widths))
                    bar_width = 0.35
                    ax3.bar(indices, widths, bar_width, label='너비 (cm)', color='blue')
                    ax3.bar(indices + bar_width, heights, bar_width, label='높이 (cm)', color='red')
                    ax3.set_xlabel('데이터 포인트', color='white')
                    ax3.set_ylabel('크기 (cm)', color='white')
                    ax3.set_title('식물 너비 및 높이', color='white')
                    ax3.legend()

                    # 피복 비율 및 식생지수 산점도
                    ax4 = self.ui.fig.add_subplot(gs[1, 1], facecolor='black')
                    ax4.scatter(ccis, vegetation_indices, alpha=0.7, color='purple')
                    ax4.set_xlabel('피복 비율', color='white')
                    ax4.set_ylabel('식생지수', color='white')
                    ax4.set_title('피복 비율 vs. 식생지수', color='white')

                    # Adjust layout
                    self.ui.fig.tight_layout()
                    self.ui.canvas.draw()
                    logging.debug("Visualized data successfully.")
                except UnicodeDecodeError as e:
                    messagebox.showerror("그래프 오류", f"그래프 레이아웃 조정 중 오류가 발생했습니다: {e}")
                    logging.error(f"Graph layout error: {e}")
                except Exception as e:
                    messagebox.showerror("그래프 오류", f"그래프를 그리는 중 오류가 발생했습니다: {e}")
                    logging.error(f"Graph plotting error: {e}")
            else:
                messagebox.showinfo("데이터 없음", "저장된 성장 데이터가 없습니다.")
                logging.info("No data found in growth_data table.")
        except Exception as e:
            messagebox.showerror("데이터 조회 오류", f"데이터를 조회하는 중 오류가 발생했습니다: {e}")
            logging.error(f"Error in visualize_data: {e}")

    def toggle_pump(self):
        if USE_ARDUINO and self.arduino:
            self.send_command_to_arduino('PUMP')
            messagebox.showinfo("급수 펌프", "급수 펌프 토글 명령을 보냈습니다.")
            logging.info("Sent 'PUMP' command to Arduino.")
        else:
            # Dummy action
            self.ui.update_status("급수 펌프 작동 (더미 데이터)")
            logging.info("Toggled pump (dummy data).")

    def toggle_led(self):
        if USE_ARDUINO and self.arduino:
            self.send_command_to_arduino('LED')
            messagebox.showinfo("LED 조명", "LED 조명 토글 명령을 보냈습니다.")
            logging.info("Sent 'LED' command to Arduino.")
        else:
            # Dummy action
            self.ui.update_status("LED 조명 토글 (더미 데이터)")
            logging.info("Toggled LED (dummy data).")

    def send_command_to_arduino(self, command):
        try:
            self.arduino.write(f"{command}\n".encode())
            logging.debug(f"Sent command to Arduino: {command}")
        except Exception as e:
            messagebox.showerror("통신 오류", f"Arduino와의 통신에 실패했습니다: {e}")
            logging.error(f"Failed to send command to Arduino: {e}")

class SmartFarmApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.logic = SmartFarmLogic()  # Initialize without UI
        self.ui = SmartFarmUI(self.root, self.logic)
        self.logic.set_ui(self.ui)  # Assign UI to logic after UI is created

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SmartFarmApp()
    app.run()
