// Pins
#define RESET 7
#define PULSE 8
#define SENSOR A1

void setup() {
    pinMode(RESET, OUTPUT);    // Pin for discharging the capacitor
    pinMode(PULSE, OUTPUT);    // Pin for turning the LED on/off
    Serial.begin(9600);
    Serial.println("off_100ms,on_200ms,on_150ms,on_100ms,on_50ms,on_20ms,on_10ms");
}

int record(int T, bool led_on) {
    digitalWrite(RESET, HIGH);        // Reset capacitor (close bypass connection)
    delayMicroseconds(100);           // Let capacitor decharge (R = 100 Ohm, C = 100 nF => tau = 10 us)
    if (led_on)
        digitalWrite(PULSE, HIGH);    // Turn LED on!
    digitalWrite(RESET, LOW);         // Start integration (open  bypass connection)
    delay(T);                         // Integrate for T milliseconds
    if (led_on)
      digitalWrite(PULSE, LOW);       // Turn LED off!
    return analogRead(SENSOR);        // Record sensor measurement
}

void loop() {
    char str[64];
    sprintf(str, "%d,%d,%d,%d,%d,%d,%d", record(100, false), record(200, true), record(150, true), 
            record(100, true), record(50, true), record(20, true), record(10, true));
    Serial.println(str);
}
