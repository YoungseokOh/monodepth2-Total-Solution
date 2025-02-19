import os
from tqdm import tqdm
from PIL import Image

def split_images(src_folder, dst_parent_folder, batch_size=100, skip_frame=False):
    # skip_frame 옵션이 켜졌다면 결과를 저장할 폴더를 새로 생성합니다.
    if skip_frame:
        dst_parent_folder = os.path.join(dst_parent_folder, "skipped")
    os.makedirs(dst_parent_folder, exist_ok=True)
    
    # 이미지 확장자를 기준으로 파일 목록을 가져옵니다.
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(extensions)]
    image_files.sort()  # 정렬하여 순서를 보장합니다.
    
    total_images = len(image_files)
    if skip_frame:
        # 건너뛰기 모드에서는 실제 저장될 이미지 수는 약 total_images/2 입니다.
        effective_total = (total_images + 1) // 2  # 홀수면 마지막 프레임 포함
        num_batches = (effective_total + batch_size - 1) // batch_size
    else:
        num_batches = (total_images + batch_size - 1) // batch_size

    for batch in tqdm(range(num_batches), desc="Processing batches"):
        batch_folder = os.path.join(dst_parent_folder, f"batch_{batch+1:03d}")
        os.makedirs(batch_folder, exist_ok=True)
        
        if skip_frame:
            # 출력 배치당 batch_size개의 이미지를 얻으려면
            # 소스에서는 2*batch_size 범위에서 2칸씩 건너뛰어 선택합니다.
            start_idx = batch * batch_size * 2
            end_idx = (batch + 1) * batch_size * 2
            batch_images = image_files[start_idx:end_idx:2]
        else:
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            batch_images = image_files[start_idx:end_idx]
        
        for idx, filename in enumerate(batch_images):
            # 출력 파일 이름은 10자리 0 패딩 숫자로 만듭니다.
            new_filename = f"{idx:010d}.jpg"
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(batch_folder, new_filename)
            
            try:
                im = Image.open(src_path)
                # JPEG는 RGB 모드를 요구하므로, 그렇지 않으면 변환합니다.
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.save(dst_path, "JPEG")
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

if __name__ == '__main__':
    src_folder = "/workspace/data/A6_rear_quater_set/ch2"
    dst_parent_folder = "/workspace/data/A6_rear_quater_set/ch2_md2_dataset"
    
    # skip_frame 옵션을 켜면 소스 이미지에서 1프레임씩 건너뛰어 저장합니다.
    # 예를 들어, 소스가 prev_00000000000.jpg, prev_00000000001.jpg, prev_00000000002.jpg 일 때,
    # 출력은 00000000000.jpg (소스의 0번째)와 00000000001.jpg (소스의 2번째)가 됩니다.
    split_images(src_folder, dst_parent_folder, batch_size=100, skip_frame=True)
