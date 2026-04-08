import torch

# 1. โหลดโมเดลตัวเดิมของคุณ (เอาตัวต้นฉบับ best_piano_model.pt มาเลยครับ)
original_path = 'best_piano_model.pt'
checkpoint = torch.load(original_path, map_location='cpu')

# ถ้า checkpoint ที่โหลดมามี 'model' อยู่แล้วให้ดึงออกมา แต่ถ้าไม่มีให้ใช้ตัวมันเองเลย
state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint

# 2. สร้างโครงสร้าง Nested Dictionary ตามที่ Library ต้องการ
# Library นี้คาดหวัง 5 ส่วนหลัก: note, pedal, reg_onset, reg_offset, velocity
new_model_struct = {
    'note_model': {},
    'pedal_model': {},
    'reg_onset_model': {},
    'reg_offset_model': {},
    'velocity_model': {}
}

# 3. จัดกลุ่ม Keys (Logic: ถ้า Key ไหนขึ้นต้นด้วยชื่อโมเดลย่อย ให้ยัดใส่กลุ่มนั้น)
for key, value in state_dict.items():
    found = False
    for sub_model in new_model_struct.keys():
        if key.startswith(sub_model + "."):
            # ตัดชื่อ prefix ออกเพื่อให้ข้างในเหลือแค่ชื่อ Layer (เช่น note_model.conv1 -> conv1)
            new_key = key.replace(sub_model + ".", "")
            new_model_struct[sub_model][new_key] = value
            found = True
            break
    
    # ถ้า Key ไหนไม่มี Prefix (กรณีโมเดลคุณเป็นแบบแบน) 
    # เราจะเหมาว่ามันคือ note_model ไว้ก่อน
    if not found:
        new_model_struct['note_model'][key] = value

# 4. แพ็กกลับลงไฟล์ใหม่
final_checkpoint = {'model': new_model_struct}
torch.save(final_checkpoint, 'best_piano_model_fixed.pt')

print("✅ สร้างไฟล์ 'best_piano_model_final.pt' สำเร็จ!")
print("ลองนำไฟล์นี้ไปใส่ใน extracter.py แทนนะครับ")