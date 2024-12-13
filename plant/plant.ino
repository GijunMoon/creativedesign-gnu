#include <MQUnifiedsensor.h>
#include <Adafruit_NeoPixel.h>
#include <avr/power.h>
#include <Wire.h>
#include <RTClib.h>
#include <DHT.h>

#define DHTPIN 5
#define ledPin 7
#define relayPin 12
#define RTCSDA A4
#define RTCSCL A5
#define RTCSQW 13
#define soilHumiditySensor A0
#define lightSensor A1
#define waterLevelSensor A2
#define PHSensor A2
#define MQ135Pin A3           

#define placa "Arduino UNO"
#define RatioMQ135CleanAir 3.6  // RS/R0 = 3.6 ppm

#define Vref 4.95
int i=0;

Adafruit_NeoPixel pixels(20, ledPin, NEO_GRB + NEO_KHZ800);
MQUnifiedsensor MQ135(placa, 5, 10, MQ135Pin, "MQ-135");  // 센서 선언
DHT dht(DHTPIN, DHT11);
RTC_DS1307 rtc;

struct DHTData {
  float humidity;
  float temperature;
};

void waterPump();//
void light(int R, int G, int B);///
int readSoilHumidity();/// 수분이 많을수록 0에 가까움
int readLight();///
int readWaterlevel();///
float readPH();//
float readAir();///
void readTime();///
void relayOn();///
void relayOff();///
DHTData readDHT();///



void setup() {
  Serial.begin(9600);           // 시리얼 포트 초기화

  pinMode(motorIN3, OUTPUT);
  pinMode(motorIN4, OUTPUT);
  pinMode(motorSpeedB, OUTPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(soilHumiditySensor, INPUT);
  pinMode(lightSensor, INPUT);
  pinMode(waterLevelSensor, INPUT);
  pinMode(PHSensor, INPUT);

  dht.begin(); // DHT 센서 초기화

// 이산화탄소 

  MQ135.setRegressionMethod(1);  // _PPM =  a*ratio^b (PPM 농도와 상수 값을 계산하기 위한 수학 모델 설정)
  MQ135.init();

  float calcR0 = 0;
  for(int i = 1; i<=10; i ++)
  {
    MQ135.update(); // Update data, the arduino will read the voltage from the analog pin
    calcR0 += MQ135.calibrate(RatioMQ135CleanAir);
  }
  MQ135.setR0(calcR0/10);
  
  MQ135.update(); // Update data, the arduino will read the voltage from the analog pin
  MQ135.setA(110.47); MQ135.setB(-2.862); // Configure the equation to calculate CO2 concentration value

//온습도
  dht.begin();

//led
  #if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
    clock_prescale_set(clock_div_1);
  #endif
    // END of Trinket-specific code.

    pixels.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)
  
  Serial.println("setup END");
//RTC
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    while (1);
  }
  if (!rtc.isrunning()) {
    Serial.println("RTC is NOT running, let's set the time!");
    // 다음 줄은 컴파일된 시점의 날짜 및 시간으로 설정합니다.
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
}

}

void loop() {////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Serial.println(readAir());
  // Serial.println(readLight());
  // Serial.println(readWaterLevel());
  // Serial.println(readSoilHumidity());
  // light(255, 255, 255);
  // readTime();
  // DHTData dhtData = readDHT();
  // Serial.println(dhtData.humidity); // %
  // Serial.println(dhtData.temperature); // *C

  int light = readLight();
  delayMicroseconds(10);
  int water = readWaterLevel();
  delayMicroseconds(10);
  int soilHumidity = readSoilHumidity();
  delayMicroseconds(10);
  float CO2 = readAir();
  delayMicroseconds(10);
  DHTData dhtData = readDHT();
  delayMicroseconds(10);

  if (light<500) {
    light(255,255,255);
  } else {
    light(0,0,0);
  }
  if (soilHumidity < 400) {
    waterPump();
  }


  delay(1000);
}

void waterPump() { //////////////// 작동안함
  Serial.println("pump");
  digitalWrite(motorIN3, HIGH);
  digitalWrite(motorIN4, LOW);
  analogWrite(motorSpeedB, 255);
  
  delay(1000);

  analogWrite(motorSpeedB, 0);
}

float readPH() {
  float sensorValue;
  int m;
  long sensorSum;
  unsigned long int avgValue;
  int buf[10];                
  relayOff();
  for(int i=0;i<10;i++)       
  { 
    buf[i]=analogRead(PHSensor);
    delay(10);
  }
  for(int i=0;i<9;i++)        //sort the analog from small to large
  {
    for(int j=i+1;j<10;j++)
    {
      if(buf[i]>buf[j])
      {
        int temp=buf[i];
        buf[i]=buf[j];
        buf[j]=temp;
      }
    }
  }
  avgValue=0;
  for(int i=2;i<8;i++)                      //take the average value of 6 center sample
  avgValue+=buf[i];
  sensorValue =   avgValue/6;
  
  return 7-1000*(sensorValue-365)*Vref/59.16/1023;
}

float readAir() {
    MQ135.update(); // Update data, the arduino will read the voltage from the analog pin

    float CO2 = MQ135.readSensor(); // Sensor will read PPM concentration using the model, a and b values set previously or from the setup

    return CO2+400;
}

void relayOn() {
  digitalWrite(relayPin, HIGH);
}

void relayOff() {
  digitalWrite(relayPin, LOW);
}

void light(int R , int G, int B) {
  pixels.clear(); // Set all pixel colors to 'off'

  for(int i=0; i<20; i++) { // For each pixel
    pixels.setPixelColor(i, pixels.Color(R,G,B));

    
  }
  pixels.show();   // Send the updated pixel colors to the hardware.
}

int readLight() {
  int light = analogRead(lightSensor);
  return light;
}

int readWaterLevel() {
  relayOn();
  delay(10);
  return analogRead(waterLevelSensor);
}

int readSoilHumidity() {
  return analogRead(soilHumiditySensor);
}
void readTime() {
  DateTime now = rtc.now();

  Serial.print(now.year(), DEC);
  Serial.print('/');
  Serial.print(now.month(), DEC);
  Serial.print('/');
  Serial.print(now.day(), DEC);
  Serial.print(" ");
  Serial.print(now.hour(), DEC);
  Serial.print(':');
  Serial.print(now.minute(), DEC);
  Serial.print(':');
  Serial.print(now.second(), DEC);
  Serial.println();
}


DHTData readDHT() {
  DHTData data;
  data.humidity = dht.readHumidity(); // 습도 읽기
  data.temperature = dht.readTemperature(); // 섭씨 온도 읽기
  return data;
}











