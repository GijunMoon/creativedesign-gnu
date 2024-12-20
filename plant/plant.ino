#include <MQUnifiedsensor.h>
#include <Adafruit_NeoPixel.h>
#include <DHT.h>

#define DHTPIN 5
#define ledPin 7
#define motorRelay 12
#define soilHumiditySensor A0
#define lightSensor A1
#define waterLevelSensor A2
#define PHSensor A4
#define MQ135Pin A3           

#define placa "Arduino UNO"
#define RatioMQ135CleanAir 3.6  // RS/R0 = 3.6 ppm

#define R_DIV 4660.0

#define Vref 4.95
int i=0;
int count = 0;

Adafruit_NeoPixel pixels(20, ledPin, NEO_GRB + NEO_KHZ800);
MQUnifiedsensor MQ135(placa, 5, 10, MQ135Pin, "MQ-135");  // 센서 선언
DHT dht(DHTPIN, DHT11);

struct DHTData {
  float humidity;
  float temperature;
};

void waterPump();//
void light(int R, int G, int B);///
int readSoilHumidity();/// 수분이 많을수록 0에 가까움
float readLight();///
int readWaterlevel();///
float readPH();//
float readAir();///
DHTData readDHT();///



void setup() {
  Serial.begin(9600);           // 시리얼 포트 초기화

  pinMode(ledPin, OUTPUT);
  pinMode(motorRelay, OUTPUT);

  pinMode(DHTPIN, INPUT);

  pinMode(soilHumiditySensor, INPUT);
  pinMode(lightSensor, INPUT);
  pinMode(waterLevelSensor, INPUT);
  pinMode(PHSensor, INPUT);

  dht.begin(); // DHT 센서 초기화
  digitalWrite(motorRelay, HIGH);
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

}

void loop() {////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  float currentLight = readLight();
  delayMicroseconds(10);
  int water = readWaterLevel();
  delayMicroseconds(10);
  int soilHumidity = readSoilHumidity();
  delayMicroseconds(10);
  float CO2 = readAir();
  delayMicroseconds(10);
  float PH = readPH();
  delayMicroseconds(10);
  DHTData dhtData = readDHT();
  delayMicroseconds(10);
  
  

  Serial.print(currentLight);
  Serial.print(",");
  Serial.print(water);
  Serial.print(",");
  Serial.print(soilHumidity);
  Serial.print(",");
  Serial.print(CO2);
  Serial.print(",");
  Serial.print(PH);
  Serial.print(",");
  Serial.print(dhtData.temperature);
  Serial.print(",");
  Serial.println(dhtData.humidity);


  if (currentLight<200) {
    light(255,255,255);
  } else {
    light(0,0,0);
  }
  if (soilHumidity  > 400) {
    count++;
    if (count > 15) {
      waterPump();
      count = 0;
    }
  } else {
    count = 0;
  }


  delay(1000);
}

void waterPump() { 
  digitalWrite(motorRelay, LOW);
  delay(1000);
  
  digitalWrite(motorRelay, HIGH);
}

float readPH() {
  float sensorValue;
  int m;
  long sensorSum;
  unsigned long int avgValue;
  int buf[10];                
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



void light(int R , int G, int B) {
  pixels.clear(); // Set all pixel colors to 'off'

  for(int i=0; i<20; i++) { // For each pixel
    pixels.setPixelColor(i, pixels.Color(R,G,B));

    
  }
  pixels.show();   // Send the updated pixel colors to the hardware.
}

float readLight() {
  int light = analogRead(lightSensor);
  float voltage = light * (5.0 / 1023.0);
  float resistance = R_DIV * ((5.0 / voltage) - 1.0);
  float lux = 500000.0 / resistance;
  return lux;
}

int readWaterLevel() {
  delay(10);
  return analogRead(waterLevelSensor);
}

int readSoilHumidity() {
  return analogRead(soilHumiditySensor);
}

DHTData readDHT() {
  DHTData data;
  data.humidity = dht.readHumidity(); // 습도 읽기
  data.temperature = dht.readTemperature(); // 섭씨 온도 읽기
  return data;
}






