import music21
import os

def midi_to_sheet(midi_path, output_xml_path):
    # 1. แปลงเส้นทางไฟล์ให้เป็นแบบเต็ม (Absolute Path)
    abs_midi_path = os.path.abspath(midi_path)
    abs_xml_path = os.path.abspath(output_xml_path)
    
    # 2. เช็คก่อนเลยว่าไฟล์มีอยู่จริงไหม
    if not os.path.exists(abs_midi_path):
        print(f"❌ Error: หาไฟล์ไม่เจอที่ {abs_midi_path}")
        print("💡 แนะนำ: ตรวจสอบว่าขั้นตอนก่อนหน้าสร้างไฟล์นี้สำเร็จไหม หรือพิมพ์ชื่อไฟล์ถูกไหมครับ")
        return # หยุดการทำงานทันที ไม่ต้องทำต่อให้เกิด Error
        
    print(f"กำลังอ่านไฟล์ MIDI จาก: {abs_midi_path}")
    
    # 3. โหลดไฟล์ MIDI 
    score = music21.converter.parse(abs_midi_path)
    
    # 4. ปรับจังหวะให้เข้าที่ (Quantization)
    score = score.quantize([4]) 
    
    # 5. แปลงและเซฟเป็นไฟล์ MusicXML
    score.write('musicxml', fp=abs_xml_path)
    print(f"✅ แปลงไฟล์สำเร็จ! บันทึกเป็น: {abs_xml_path}")

# สั่งทำงาน
if __name__ == "__main__":
    # ใส่ชื่อไฟล์ตรงๆ แบบนี้ได้เลย (เดี๋ยว os.path.abspath จัดการต่อให้)
    midi_to_sheet("output.mid", "output_sheet.xml")