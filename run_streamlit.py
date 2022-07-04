from libs import *
from utils_func import create_dir, main_processing



create_dir("tempDir")

def load_image(image_file):
	img = Image.open(image_file)
	return img


def streamlit_app():
    detection_model_path = "weight_files/clothes_detection_model.pt"
    background_model_path = "weight_files/model.h5"
    # save_path = ""
    image_file = None
    st.title("""WELCOME TO MY APP""")
    st.subheader("""FOR BACKGROUND REMOVAL AND CHANGE!""")
    col1 = None
    col2 = None
    final_img = None
    with st.spinner("[UPLOAD] Image uploading"):
        try:
            image_file = st.file_uploader('[UPLOAD] Please upload your image:', type=["png", "jpg", "jpeg"])
            time.sleep(1)
        except:
            print("[ERROR] Sorry, something went wrong!")
            pass
    # print(type(image_file))

    if image_file is not None:
        st.success("Load image successfully!...")
        image = load_image(image_file)
        # print(type(image))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Image before processing")
        save_path = "tempDir\\"+ image_file.name
        image.save(save_path)


        image_path, details = save_path, image_file

        if details is not None:
            with col2:
                with st.spinner("[PROCESSING] Image processing"):
                    final_img_path = main_processing(col1, col2, col3, sport_bg_path=stadium_sport_bg_path, swim_bg_path=beach_swim_bg_path,
                                          office_bg_path=office_bg_path, img_path=image_path, name=details.name,
                                          detection_model_path=detection_model_path,
                                          background_model_path=background_model_path)
                    time.sleep(1)

            with col1:
                if final_img_path is not None:
                        final_img = load_image(final_img_path)
                        st.image(final_img, caption="Image after processing")
                        st.balloons()
                        with col2:
                            with open(final_img_path, "rb") as file:
                                st.write('\n')
                                st.write('\n')
                                st.write('\n')
                                st.write('\n')
                                st.write('\n')

                                file_name = final_img_path.split("\\")[-1].split(".")[-2]
                                if st.download_button(
                                    label="Download postprocessing image",
                                    data=file,
                                    file_name= file_name,
                                    mime="image/png"
                                ):
                                    st.success('[DOWNLOAD] Download sucessfully!')



if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

    bg_path = ""
    background_model_path = "weight_files\\model.h5"
    detection_model_path = "weight_files\\clothes_detection_model.pt"

    stadium_sport_bg_path = "backgrounds\\camnou_stadium.jpg"
    beach_swim_bg_path = "backgrounds\\beach.jpg"
    office_bg_path = "backgrounds\\office-bg.jpg"

    image_path = None

    streamlit_app()
