from django import forms


class UploadImageForm(forms.Form):
    image = forms.ImageField(label='上传图片文件', required=True)
