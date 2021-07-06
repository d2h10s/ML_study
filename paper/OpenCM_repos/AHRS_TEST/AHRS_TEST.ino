#define COM             0x2C // ,
#define SOL             0x2A // *
#define EOL_CR          0x0D
#define EOL_LF          0x0A
#define AHRS_DATA_SIZE  4
float data[4];
char buf[64];
uint8_t buf_idx = 0;

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200);
  while(!Serial);
  Serial2.println("<sor0>");
  delay(1000);
  while (Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<sof1>");
  delay(1000);
  while (Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<sog0>");
  delay(1000);
  while (Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<sot1>");
  delay(1000);
  while (Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<soa4>");
  delay(1000);
  while (Serial2.available()) Serial.write(Serial2.read());
  Serial.println("Program start");
}

void loop() {
  /*
  if(!getEulerAngle()) Serial.println("read failed");
  //if (Serial2.available()) Serial.write(Serial2.read());
  if (Serial.available()) Serial2.write(Serial.read());
  delay(1000);
  */
  
  Serial2.write(SOL);
  while(Serial2.available()){
    Serial.write(Serial2.read());
  }
}

int getEulerAngle() {
  Serial2.write(SOL);
  while (Serial2.available()) {
    buf[buf_idx] = Serial2.read();
    buf_idx++;

    if (buf[buf_idx - 1] == EOL_LF) buf[buf_idx - 1] = ',';

    if (buf[buf_idx - 1] == SOL){
      char *seg = strtok(buf, ",");

      for (int i = 0; i < AHRS_DATA_SIZE; i++){
        data[i] = atof(seg);
        seg = strtok(NULL, ",");
      }
      buf_idx = 0;
    }
  }
  Serial.println(data[0]);;
  return 1;
}
