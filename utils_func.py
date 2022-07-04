from metrics import dice_loss, dice_coef, iou
from libs import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Global parameters """
H = 512
W = 512


def load_image(image_file):
	img = Image.open(image_file)
	return img


def choose_background(col1, col2, col3):
    background_file = None
    bg_image = None
    with col3:
        with st.spinner("[UPLOAD] Background uploading"):
            try:
                if background_file is None:
                    try:
                        background_file = st.file_uploader('[UPLOAD] Please upload your background:', type=["png", "jpg", "jpeg"])
                        time.sleep(1)
                    except:
                        pass
            except:
                print("[ERROR] Sorry, something went wrong!")
                pass

    if background_file is not None:
        with col2:
            st.success("Load background successfully!...")

        bg_image = load_image(background_file)
        print(type(bg_image))
        save_path = "backgrounds\\" + background_file.name
        bg_image.save(save_path)

        return bg_image, save_path


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_clothe(model_path, img_path):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)  # or yolov5n - yolov5x6, custom

    # Inference
    results = model(img_path)

    # Results
    new_results = results.pandas().xyxy[0].sort_values("confidence").to_json(orient="records")
    new_results = json.loads(new_results)

    classes_in_img = []
    classes_count_dict = {}
    item = ''
    if len(new_results) != 0:
        for i in range(len(new_results)):
            classes_in_img.append(new_results[i]['name'])
        set_of_classes = set(classes_in_img)
        list_of_classes = list(set_of_classes)

        older_value_count = 0
        for idx in range(len(list_of_classes)):
            value_count = classes_in_img.count(list_of_classes[idx])
            if value_count > older_value_count:
                item = list_of_classes[idx]

    else:
        item = None

    return item



def main_processing(col1, col2, col3, sport_bg_path, swim_bg_path, office_bg_path, img_path, name, detection_model_path, background_model_path):
    """ Seeding """
    bg_path = None
    np.random.seed(42)
    tf.random.set_seed(42)

    model_path = detection_model_path

    stadium_sport_bg_path = sport_bg_path
    beach_swim_bg_path = swim_bg_path
    office_bg_path = office_bg_path

    """ Directory for storing files """
    create_dir("remove_bg")

    st.write('Auto detect or choosing background? ')

    if bg_path is None:
        if st.checkbox('Choose background'):
            try:
                bg_img, save_path = choose_background(col1, col2, col3)
                bg_path = save_path
            except:
                pass

            """ Directory for storing files """
        elif st.checkbox('Automatic background'):
            item = check_clothe(model_path, img_path)
            if item == 'sport':
                bg_path = stadium_sport_bg_path
                st.write("Hãy tiếp tục luyện tập TDTT chăm chỉ nhé!...")
            if item == 'swim':
                bg_path = beach_swim_bg_path
                st.write("Thời tiết thế này không đi biển hơi phí nhé!...")
            if item == 'office':
                bg_path = office_bg_path
                st.write("Chơi nhiều roài, đi làm chăm chỉ thuii...")
            if item == None:
                st.warning("Sorry, mô hình chúng tôi không biết bạn đang mặc cái quái gì hết...")
                st.warning("Chọn background bạn muốn nhé!")
                try:
                    background_img, save_path = choose_background(col1, col2, col3)
                    bg_path = save_path
                except:
                    pass

            else:
                pass

        else:
            pass

    if bg_path is not None:
        """ Loading model: DeepLabV3+ """
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(background_model_path)

        """ Read the image """
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)
        y = y > 0.5

        photo_mask = y
        background_mask = np.abs(1 - y)
        cv2.imwrite(
            f"remove_bg\\{name}_1.png",
            photo_mask * 255)
        cv2.imwrite(
            f"remove_bg\\{name}_2.png",
            background_mask * 255)

        cv2.imwrite(
            f"remove_bg\\{name}_3.png",
            image * photo_mask)
        cv2.imwrite(
            f"remove_bg\\{name}_4.png",
            image * background_mask)

        bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        print(bg_img.shape)
        new_bg_img = cv2.resize(bg_img, (w, h))

        new_new_bg_img = new_bg_img * background_mask

        cv2.imwrite(
            f"remove_bg\\{name}_5.png",
            new_new_bg_img)

        final_final_img = new_new_bg_img + image * photo_mask

        final_img_path = f"remove_bg\\{name}_6.png"
        cv2.imwrite(
            final_img_path,
            final_final_img)

        return final_img_path


# if __name__ == '__main__':
#     """ Seeding """
#     np.random.seed(42)
#     tf.random.set_seed(42)
#
#     bg_path = ""
#     background_model_path = "weight_files\\model.h5"
#     detection_model_path = "weight_files\\clothes_detection_model.pt"
#
#     stadium_sport_bg_path = "backgrounds\\camnou_stadium.jpg"
#     beach_swim_bg_path = "backgrounds\\beach.jpg"
#     office_bg_path = "backgrounds\\office-bg.jpg"
#
#     img_path = "images\\truong-van-bang-10163832.jpg"
#
#     main(sport_bg_path=stadium_sport_bg_path, swim_bg_path=beach_swim_bg_path, office_bg_path=office_bg_path, name="a", img_path=img_path, detection_model_path=detection_model_path, background_model_path=background_model_path)