# Project-I
In Project I, I perform the license plate recognition process by extracting the features of the image with the Green Parking dataset
(https://thigiacmaytinh.com/tai-nguyen-xu-ly-anh/tong-hop-data-xu-ly-anh/?fbclid=IwAR2tajA5Ku83kIrb09ovhmb_68Zmdwo9KvV_CSNBCTbuIIsiK_FUM4W4Dh8)

In this problem, I separate the area containing the license plate and separate each character in it for the purpose of performing feature recognition.

With input image:

![image](https://user-images.githubusercontent.com/68015472/148957498-66227116-63a6-4f3c-9d29-a43810df50a6.png)

And the results after covering the number plate area and removing the inappropriate areas by Histogram:

![image](https://user-images.githubusercontent.com/68015472/148957751-733b1ab0-e126-4fbd-8afe-922e590e5cce.png)

![image](https://user-images.githubusercontent.com/68015472/148958054-aca76d6b-ca04-451c-9569-602628edbca3.png)

Area 1

![image](https://user-images.githubusercontent.com/68015472/148958068-e11a4371-eef9-4721-bdac-5794cf3c6223.png)

Area 2

Because the characters in the license plate in Vietnam have a fixed height and width size. Therefore, the characters were separated after applying the height and height condition:

![image](https://user-images.githubusercontent.com/68015472/148958654-ef798430-7666-4243-bc4f-da9cbd2eae11.png)

To be able to identify by features of the image, I perform a character-by-character conversion from a 2D image to a 1D image representing the signal of the image from top to bottom:

![image](https://user-images.githubusercontent.com/68015472/148958976-247c40c0-1b7c-4e3d-a0cc-aa12167e0947.png)
![image](https://user-images.githubusercontent.com/68015472/148958996-054a216f-2586-42b8-bbb9-9d5c4da482ab.png)

![image](https://user-images.githubusercontent.com/68015472/148959430-ff84d6ea-928b-4de4-ab9e-34b21aee7ca8.png)
![image](https://user-images.githubusercontent.com/68015472/148959481-703e12cf-94b5-4bce-a71a-9044ed6324c4.png)

![image](https://user-images.githubusercontent.com/68015472/148959237-51db7a10-22a6-40dc-9a9b-3f46791f6be9.png)
![image](https://user-images.githubusercontent.com/68015472/148959263-370d663e-288d-4c26-b640-4e2bfb3e91b7.png)

![image](https://user-images.githubusercontent.com/68015472/148959138-a78d67ab-7843-4d28-8629-1e1a86d6b7b0.png)
![image](https://user-images.githubusercontent.com/68015472/148959162-60b817bd-5695-4a52-8f62-1eecc40469b3.png)

![image](https://user-images.githubusercontent.com/68015472/148959193-ebf68950-e3fc-4ab8-bbf5-87500a0f410c.png)
![image](https://user-images.githubusercontent.com/68015472/148959215-81ea2925-cec8-4b7a-9cad-5f63f27c2335.png)

Based on the properties of the graph, maximum and minimum, we can select the basic features of the image and perform an extraction of those features. After completing the construction of the composite features after 300 images, I obtained an accurate recognition result of 83.83%, the correct recognition rate of alphanumeric characters.

![image](https://user-images.githubusercontent.com/68015472/148960026-2ecbda51-ef9f-4406-8a53-aa05e805dfe6.png)

![image](https://user-images.githubusercontent.com/68015472/148960050-cadad0c0-b23d-4a2e-b869-19aa99a0a664.png)

![image](https://user-images.githubusercontent.com/68015472/148960069-302eb200-b2bc-42cd-92d4-330bb95c34f9.png)






