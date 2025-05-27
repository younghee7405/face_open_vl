
from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경변수 불러오기

app = Flask(__name__)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_point(landmarks, index, w, h):
    lm = landmarks.landmark[index]
    return int(lm.x * w), int(lm.y * h)

def draw_text_on_image(img_path, text, output_path='static/with_text.jpg'):
    font_path = "./static/NanumGothic.ttf"
    font = ImageFont.truetype(font_path, 20)
    cv_img = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    lines = text.split('\n')
    x, y = 20, 20
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += 30

    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_img)
    return output_path

def refine_with_gpt(messages: list):
    prompt = "다음은 얼굴을 분석한 관상 결과야. 자연스럽고 매끄러운 말투로 요약해줘:\n\n"
    for line in messages:
        prompt += f"- {line}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filepath = 'static/result.jpg'
    file.save(filepath)

    image = cv2.imread(filepath)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    results_text, extended_results = [], []

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            forehead_y = get_point(landmarks, 10, w, h)[1]
            nose_y = get_point(landmarks, 1, w, h)[1]
            left_eye = get_point(landmarks, 33, w, h)
            right_eye = get_point(landmarks, 263, w, h)
            eye_gap = abs(right_eye[0] - left_eye[0])
            eye_width = abs(get_point(landmarks, 133, w, h)[0] - left_eye[0])
            jaw_width = abs(get_point(landmarks, 234, w, h)[0] - get_point(landmarks, 454, w, h)[0])

            if forehead_y < nose_y - 40:
                results_text.append("🧠 이마가 높고 넓어 리더십과 판단력이 뛰어납니다.")
            else:
                results_text.append("🤔 이마가 낮은 편이라 실용적이고 행동 중심적인 성향입니다.")

            if eye_gap > eye_width * 1.5:
                results_text.append("👀 눈 사이가 넓어 독립적이고 자기 주장이 강한 편입니다.")
            else:
                results_text.append("👀 눈 사이가 적당해 조화로운 대인 관계를 잘 맺습니다.")

            if jaw_width > w * 0.5:
                results_text.append("💪 턱이 각지고 넓은 편이라 책임감이 강한 성격입니다.")
            else:
                results_text.append("😊 턱이 갸름한 편이라 감성적이고 섬세한 성향입니다.")

            left_eye_start = get_point(landmarks, 33, w, h)
            left_eye_end = get_point(landmarks, 133, w, h)
            mouth_left = get_point(landmarks, 61, w, h)
            mouth_right = get_point(landmarks, 291, w, h)
            nose_top = get_point(landmarks, 6, w, h)
            nose_tip = get_point(landmarks, 1, w, h)

            if left_eye_end[1] < left_eye_start[1] - 5:
                extended_results.append("👁 눈꼬리가 올라가 있어 활발하고 낙천적인 성격입니다.")
            elif left_eye_end[1] > left_eye_start[1] + 5:
                extended_results.append("👁 눈꼬리가 내려가 있어 온순하고 차분한 인상입니다.")
            else:
                extended_results.append("👁 눈꼬리가 수평으로 균형 잡힌 인상입니다.")

            if mouth_right[1] < mouth_left[1] - 5:
                extended_results.append("😊 입꼬리가 올라가 밝고 긍정적인 성향입니다.")
            elif mouth_right[1] > mouth_left[1] + 5:
                extended_results.append("😐 입꼬리가 살짝 내려가 조용하고 신중한 스타일입니다.")
            else:
                extended_results.append("🙂 입꼬리가 중립적이며 차분한 성격입니다.")

            if nose_top[1] < nose_tip[1] - 20:
                extended_results.append("👃 콧대가 높아 자존감과 자신감이 강한 성향입니다.")
            else:
                extended_results.append("👃 콧대가 낮아 겸손하고 조화로운 성격입니다.")

    refined_text = refine_with_gpt(results_text + extended_results)
    draw_text_on_image(filepath, refined_text)
    return render_template('index.html', filename='with_text.jpg', final_analysis=refined_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
