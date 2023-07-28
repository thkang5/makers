from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import io
import boto3

AWS_ACCESS_KEY = ""#AWS_ACCESS_KEY와 AWS_SECRET_KEY에는 사용자의 AWS 액세스 키와 시크릿 액세스 키를 입력
AWS_SECRET_KEY = ""
BUCKET_NAME = ""#BUCKET_NAME에는 사용자가 업로드한 이미지를 저장할 버킷의 이름을 입력

s3 = boto3.client('s3',
        aws_access_key_id = AWS_ACCESS_KEY,
        aws_secret_access_key = AWS_SECRET_KEY)#boto3를 사용하여 S3 클라이언트를 생성

app = Flask(__name__)#Flask 애플리케이션을 생성
# export model
model_female = load_model('model_fashion.h5')
model_male = load_model('model_fashion.h5')

@app.errorhandler(404)#404 오류 발생 시, 404.html 템플릿을 렌더링
def page_not_found(error):

   return render_template('404.html'), 404

@app.route('/', methods=['GET', 'POST'])#두 가지 메서드(GET과 POST)에 대한 / 루트 경로를 처리
def index():
    if request.method == 'GET':#GET 요청의 경우, index.html 템플릿을 렌더링하여 사용자에게 보여준다
        
        return render_template('index.html')

    if request.method == 'POST':#POST 요청의 경우, 사용자가 업로드한 이미지를 받아와서 성별을 예측
        try:
            if request.form.get('gender'):

                # female predict// 파일전송까지는 제대로 된거 확인함
                img = request.files["file"]#여기까지 ㅇㅋ

                pred_img = img.read()
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다여기까지 ㅇㅋ

                #upload_img(img)
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                pred_img = Image.open(io.BytesIO(pred_img)).convert("RGB")
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                pred_img = pred_img.resize((256, 256))
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                pred_img = img_to_array(pred_img)
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                pred_img = pred_img.reshape((1, 256, 256, 3))
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                pred = model_female.predict(pred_img)
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                label = pred.argmax()
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                label = 'f' + str(label)
                #return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                                #           return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

            
                return render_template("index.html", label=label)    #성별이 여성일 경우, 이미지를 읽고 전처리한 뒤 예측 모델을 사용하여 성별을 예측
        
            else:
                # male predict

                img = request.files["file"]
                pred_img = img.read()
                #upload_img(img)
                pred_img = Image.open(io.BytesIO(pred_img)).convert("RGB")
                pred_img = pred_img.resize((256, 256))
                pred_img = img_to_array(pred_img)
                pred_img = pred_img.reshape((1, 256, 256, 3))
                pred = model_male.predict(pred_img)
                label = pred.argmax()
                label = 'm' + str(label)
               #            return render_template('404.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다

                return render_template("index.html", label=label)    #성별이 남성일 경우, 동일한 과정을 거친 뒤 예측 결과에 'm'을 붙여 label 변수에 할당
        except:

           return render_template('index.html'), 404#예외가 발생한 경우, 404.html 템플릿을 렌더링하여 404 오류를 보여준다




def upload_img(image):#upload_img 함수는 이미지를 AWS S3 버킷에 업로드하는 역할
    image.seek(0) # s3 저장에 필요   파일 포인터를 맨 앞으로 이동시켜야 한다는 의미
    s3.put_object(#s3.put_object를 사용하여 이미지를 S3 버킷에 저장
        Bucket = BUCKET_NAME, # 버킷 이름
        Body = image, # 업로드 파일
        Key = 'image/{}'.format(image.filename), # 저장 위치 및 이름 지정
        ContentType = image.content_type) # 이미지 타입
    

if __name__ == "__main__":#애플리케이션을 실행합니다. debug=True로 설정하면 디버그 모드로 실행되어 코드 수정 시 자동으로 애플리케이션이 재시작
    app.run(debug=True)