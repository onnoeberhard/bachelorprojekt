/* 
    Nephelometer Steuerung
    Bachelorprojekt Brinkmann, Eberhard, Stock 2020
*/

// Pins
#define RESET 7
#define PULSE 8
#define SENSOR A1

// Integrationszeit [ms]
#define T 100

void setup() {
    pinMode(RESET, OUTPUT);    // Pin for discharging the capacitor
    pinMode(PULSE, OUTPUT);    // Pin for turning the LED on/off
    Serial.begin(9600);
}

int record(bool led_on) {
    digitalWrite(RESET, HIGH);        // Reset capacitor (close bypass connection)
    delayMicroseconds(100);           // Let capacitor discharge (R = 100 Ohm, C = 100 nF => RC = 10 us)
    digitalWrite(RESET, LOW);         // Start integration (open  bypass connection)
    if (led_on)
        digitalWrite(PULSE, HIGH);    // Turn LED on!
    delay(T);                         // Integrate for T milliseconds
    if (led_on)
      digitalWrite(PULSE, LOW);       // Turn LED off!
    return analogRead(SENSOR);        // Record sensor measurement
}

void loop() {
    // Record time to adjust delay for next measurement
    unsigned long time_start = millis();

    // Make the measurements
    int led_off = record(false);    // Background measurement
    int led_on  = record(true);     // Measurement with LED on

    // Print result to Serial
    char str[64];
    sprintf(str, "%010lu,%04d,%04d", time_start, led_off, led_on);
    Serial.println(str);
    
    // Repeat exactly every 500ms
    unsigned long time_diff = (millis() - time_start);
    delay(500 - time_diff);
}
