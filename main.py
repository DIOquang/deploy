import os
import json

# Đường dẫn mặc định trên Lightning AI
DATA_DIR = "/teamspace/studios/this_studio/hf_dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.jsonl")

# Số lượng ảnh mục tiêu (19 vật phẩm x 6 chất liệu x 500 biến thể)
EXPECTED_COUNT = 57000 

def check_data_integrity():
    print("🔍 BẮT ĐẦU KIỂM TRA TOÀN VẸN DỮ LIỆU...\n")

    # 1. Kiểm tra số lượng ảnh thực tế
    actual_images_count = 0
    if os.path.exists(IMAGES_DIR):
        # Chỉ đếm các file có đuôi .png
        image_files = set(f for f in os.listdir(IMAGES_DIR) if f.endswith('.png'))
        actual_images_count = len(image_files)
        print(f"📸 Số lượng ảnh thực tế trong 'images/': {actual_images_count:,} / {EXPECTED_COUNT:,}")
    else:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy thư mục ảnh tại {IMAGES_DIR}")
        return

    # 2. Kiểm tra số lượng prompt trong file metadata
    actual_metadata_count = 0
    metadata_image_names = set()
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                actual_metadata_count += 1
                try:
                    data = json.loads(line)
                    # Lấy tên file từ metadata (VD: "images/sword_gold_0001.png" -> "sword_gold_0001.png")
                    filename = os.path.basename(data["file_name"])
                    metadata_image_names.add(filename)
                except Exception as e:
                    pass
        print(f"📄 Số lượng dòng prompt trong 'metadata.jsonl': {actual_metadata_count:,} / {EXPECTED_COUNT:,}")
    else:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file {METADATA_PATH}")
        return

    # 3. Đánh giá sự đồng bộ (Cross-check)
    print("\n⚙️ ĐANG ĐỐI CHIẾU SỰ ĐỒNG BỘ GIỮA ẢNH VÀ METADATA...")
    missing_in_folder = metadata_image_names - image_files
    missing_in_metadata = image_files - metadata_image_names

    print("\n==================================================")
    if actual_images_count == EXPECTED_COUNT and actual_metadata_count == EXPECTED_COUNT and not missing_in_folder:
        print("✅ TUYỆT VỜI! Dữ liệu đã được tải lên và giải nén ĐẦY ĐỦ 100%.")
        print("✅ Ảnh và Metadata hoàn toàn khớp nhau. Bạn có thể BẮT ĐẦU TRAINING!")
    else:
        print("⚠️ CẢNH BÁO: Quá trình tải lên/giải nén chưa hoàn tất hoặc bị lỗi mạng.")
        if actual_images_count < EXPECTED_COUNT:
            print(f"   -> Thư mục ảnh đang thiếu {EXPECTED_COUNT - actual_images_count:,} tấm.")
        if actual_metadata_count < EXPECTED_COUNT:
            print(f"   -> File metadata đang thiếu {EXPECTED_COUNT - actual_metadata_count:,} dòng.")
        if missing_in_folder:
            print(f"   -> Có {len(missing_in_folder):,} ảnh có trong metadata nhưng KHÔNG CÓ trong thư mục.")
    print("==================================================")

if __name__ == "__main__":
    check_data_integrity()