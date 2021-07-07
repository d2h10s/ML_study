#define COM             0x2C // ,
#define SOL             0x2A // *
#define EOL_CR          0x0D
#define EOL_LF          0x0A
#define AHRS_DATA_SIZE  4
#define BAUDRATE 115200
float data[4];
char buf[64];
uint8_t buf_idx = 0;

void setup() {
  Serial.begin(115200);
  while(!Serial);
  ahrs_init();
  Serial.println("loop start");
}

void loop() {
  Serial2.write(SOL);
  while(Serial2.available()){
    Serial.write(Serial2.read());
  }
}
void ahrs_init(){
  pinMode(6, OUTPUT);
  digitalWrite(6, 0);
  delay(500);
  digitalWrite(6, 1);
  delay(500);
  Serial2.begin(BAUDRATE);
  Serial2.println("<sor0>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sof1>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sog0>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sot1>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<soa4>");
  delay(1000);
  while (Serial2.available()) Serial2.read();

}
