/*
  Purple Team - SEN0240 three-channel RAW binary recorder

  Connections
    Sensor 1 analog output -> A0
    Sensor 2 analog output -> A1
    Sensor 3 analog output -> A2
    All sensor VCC         -> Arduino 5V
    All sensor GND         -> Arduino GND

  Acquisition
    - 1,000 sample groups/second
    - Each group contains one settled 10-bit reading from A0, A1 and A2
    - Timer1 supplies the 1 ms sampling timeline
    - No digital filter, RMS, normalization, clipping or smoothing
    - 500,000 baud binary output

  Packet format (14 bytes, little endian)
    [0]      0xA5 sync byte 1
    [1]      0x5A sync byte 2
    [2..5]   uint32 sample_index
    [6..7]   uint16 A0
    [8..9]   uint16 A1
    [10..11] uint16 A2
    [12..13] CRC-16/CCITT-FALSE over bytes [2..11]

  The first ADC result after every channel switch is discarded. Therefore
  six ADC conversions are performed per 1 ms group. If acquisition or serial
  transfer overruns, an old group is not sampled late; sample_index jumps so
  the PC can detect the missing group.
*/

#include <Arduino.h>
#include <util/atomic.h>

constexpr uint8_t CHANNEL_COUNT = 3;
constexpr uint8_t EMG_PINS[CHANNEL_COUNT] = {A0, A1, A2};

constexpr uint32_t SAMPLE_RATE_HZ = 1000UL;
constexpr uint32_t SERIAL_BAUD = 500000UL;
constexpr uint8_t PACKET_SIZE = 14;
constexpr uint8_t SYNC_1 = 0xA5;
constexpr uint8_t SYNC_2 = 0x5A;

volatile uint32_t timerTicks = 0;

ISR(TIMER1_COMPA_vect) {
  ++timerTicks;
}

void configureTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // 16 MHz / 8 prescaler / (1999 + 1) = 1,000 Hz.
  OCR1A = (F_CPU / 8UL / SAMPLE_RATE_HZ) - 1UL;
  TCCR1B |= _BV(WGM12);  // CTC mode
  TCCR1B |= _BV(CS11);   // prescaler 8
  TIMSK1 |= _BV(OCIE1A);
  interrupts();
}

uint16_t readSettledAdc(uint8_t pin) {
  (void)analogRead(pin);  // discard after changing the ADC multiplexer
  return (uint16_t)analogRead(pin);
}

void writeUint16LE(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value & 0xFFU);
  destination[1] = (uint8_t)(value >> 8U);
}

void writeUint32LE(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value & 0xFFUL);
  destination[1] = (uint8_t)((value >> 8UL) & 0xFFUL);
  destination[2] = (uint8_t)((value >> 16UL) & 0xFFUL);
  destination[3] = (uint8_t)((value >> 24UL) & 0xFFUL);
}

uint16_t crc16CcittFalse(const uint8_t *data, uint8_t length) {
  uint16_t crc = 0xFFFFU;
  for (uint8_t index = 0; index < length; ++index) {
    crc ^= (uint16_t)data[index] << 8U;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000U)
          ? (uint16_t)((crc << 1U) ^ 0x1021U)
          : (uint16_t)(crc << 1U);
    }
  }
  return crc;
}

void buildPacket(
    uint8_t packet[PACKET_SIZE],
    uint32_t sampleIndex,
    const uint16_t values[CHANNEL_COUNT]) {
  packet[0] = SYNC_1;
  packet[1] = SYNC_2;
  writeUint32LE(packet + 2, sampleIndex);
  writeUint16LE(packet + 6, values[0]);
  writeUint16LE(packet + 8, values[1]);
  writeUint16LE(packet + 10, values[2]);
  const uint16_t crc = crc16CcittFalse(packet + 2, 10);
  writeUint16LE(packet + 12, crc);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  analogReference(DEFAULT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  for (uint8_t channel = 0; channel < CHANNEL_COUNT; ++channel) {
    pinMode(EMG_PINS[channel], INPUT);
  }

  // Let the USB serial connection and sensor amplifiers settle after reset.
  delay(1000);
  configureTimer1();
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  static uint32_t lastServicedTick = 0;

  uint32_t currentTick;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    currentTick = timerTicks;
  }
  if (currentTick == lastServicedTick) {
    return;
  }

  // Use only the newest scheduled tick. Any missed tick becomes a visible
  // sample_index gap instead of a late burst of falsely timed samples.
  lastServicedTick = currentTick;
  const uint32_t sampleIndex = currentTick - 1UL;

  uint16_t values[CHANNEL_COUNT];
  for (uint8_t channel = 0; channel < CHANNEL_COUNT; ++channel) {
    values[channel] = readSettledAdc(EMG_PINS[channel]);
  }

  uint8_t packet[PACKET_SIZE];
  buildPacket(packet, sampleIndex, values);

  // Never block the next acquisition for serial output. If the UART buffer
  // is unexpectedly full, omit this packet and expose the loss by index gap.
  if (Serial.availableForWrite() >= PACKET_SIZE) {
    Serial.write(packet, PACKET_SIZE);
  }
}
