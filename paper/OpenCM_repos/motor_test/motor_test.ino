#include <DynamixelWorkbench.h>
DynamixelWorkbench wb;

//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define BAUDRATE          115200  // motor and serial baudrate
#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX64_ID           2
#define SAMPLING_TIME     25     // milli second
#define MX106_CW_POS      2200
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200

//FOR PC COMMUNICATION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define STX               0x02  // start of text
#define NLF               0x0A  // end of text
#define ACK               0x06  // Acknowlegement
#define NAK               0x15  // Negative Acknowledgement

#define ACQ               0x04  // pc acquires ovservation data
#define RST               0x05  // pc commands reset environment
#define GO_CW             0x70  // pc commands MX106 goes to min position
#define GO_CCW            0x71  // pc commands MX106 goes to max position

#define RX_BUF_SIZE       128
#define TX_BUF_SIZE       128

//FOR AHRS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SOL               0x2A
#define EOL_CR            0x0D
#define EOL_LF            0x0A
#define AHRS_BUF_SIZE     64
#define AHRS_DATA_SIZE    7

//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
char ahrs_buf[AHRS_BUF_SIZE]      = {0};
float ahrs_data[AHRS_DATA_SIZE]   = {0};
int32_t temp_buf[3]               = {0};
int32_t pos_buf[3]                = {0};
int32_t vel_buf[3]                = {0};
int8_t ahrs_temp                  = 0;
uint8_t ahrs_buf_idx              = 0;
uint8_t rx_buf[RX_BUF_SIZE]       = {0};
char tx_buf[TX_BUF_SIZE]          = {0};

uint8_t command                   = 0;
bool is_MX106_on                  = false;
bool isOnline                     = false;


//MAIN PROGRAM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void setup() {
  Serial.begin(BAUDRATE);
  Serial.setTimeout(180*1000);
  while(!Serial);
  
  while (!motor_init());
}

void loop() {
  
  wb.goalPosition(MX106_ID, MX106_CW_POS);
  delay(1000);
  wb.goalPosition(MX106_ID, MX106_CCW_POS);
  delay(1000);
}


int motor_init(){
  const char* log;
  is_MX106_on = wb.init(SERIAL_DEVICE, BAUDRATE, &log);
  if (!is_MX106_on) {
    Serial.print("@Port Open Failed!");
    return 0;
  }
  
  is_MX106_on = wb.ping(MX106_ID, &log);
  if (!is_MX106_on) {
   Serial.print("@Ping test Failed!");
    return 0;
  }
  
  is_MX106_on = wb.currentBasedPositionMode(MX106_ID, MX106_CURRENT, &log);
  if (!is_MX106_on) {
    Serial.print("@set mode Failed!");
    return 0;
  }
  
  return 1;
}
