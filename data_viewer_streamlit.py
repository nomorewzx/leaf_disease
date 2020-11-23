import streamlit as st
import os
import pandas as pd
import cv2

LABEL_IDX_NAME_MAPPING = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

DATA_DIR = './cassava-leaf-disease-classification-data'


def main():
    st.header('Casseva Leaf Disease Detection')
    st.markdown("""There are 5 different classes, including 4 disease labels and ** healthy ** label:
    
                    1. Cassava Bacterial Blight (CBB)
                    2. Cassava Brown Streak Disease (CBSD)
                    3. Cassava Green Mottle (CGM)
                    4. Cassava Mosaic Disease (CMD)
                    5. Healthy
                """)

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_df.loc[:, 'label_name'] = train_df['label'].apply(lambda label_idx: LABEL_IDX_NAME_MAPPING[label_idx])

    selected_label = st.selectbox('Select a class to view', list(LABEL_IDX_NAME_MAPPING.values()), 0)

    # Select a label to view
    selected_df = train_df[train_df['label_name'] == selected_label]

    if selected_df is None or selected_df.shape[0] == 0:
        st.write('No corresponding label images found in train dataset')
        return None

    selected_idx = st.slider('Choose an image to view', 0, selected_df.shape[0] - 1, 0)
    selected_df = selected_df.reset_index(drop=True)
    selected_img_id = selected_df.loc[selected_idx, 'image_id']
    st.write('** {} **'.format(selected_img_id))
    # Read and display image
    image_path = os.path.join(os.path.abspath(DATA_DIR), 'train_images', selected_img_id)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    st.image(img)

    st.write('### The number of images per class ###')
    class_num_df = train_df.groupby(by=['label_name']).agg({'image_id': 'count'})
    st.bar_chart(class_num_df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()