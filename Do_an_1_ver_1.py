import cv2
import matplotlib.pyplot as plt
import statistics
import numpy as np
import math
import time

# Nhập dữ liệu đầu vào
path = 'GreenParking/'
# Đọc dữ liệu chứa tên của từng ảnh
f = open('GreenParking/location.txt', 'r')
lst_1 = [f.read().splitlines()]
lst_2 = []
image_list = []
count = 0
# Lưu lại tên của từng file ảnh
for i in lst_1[0]:
    lst_2.append(i.split()[0])

'''
Đọc từng ảnh được lưu trong lst_2
'''


def image_input(i):
    path_img = path + lst_2[i]
    image_original = cv2.imread(path_img)
    # Trả về ảnh gốc
    return image_original


'''
xư ly hinh anh de contour duoc bien so xe
'''


def image_process(im, img):
    img_cut = img.copy()
    img_detect = img.copy()
    count_img = 0
    # xử lý làm mịn ảnh xóa đi nhiễu bằng hàm gauss
    im_gauss = cv2.GaussianBlur(im, (5, 5), 0)
    im_binary = cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -8)
    plt.imshow(im_binary)
    plt.show()
    contours, hierachy = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if 0.9 < w / h < 1.4 and 55 < w < 100 and 40 < h < 100:
            count_img += 1
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            img_cnt = img_cut[y:y + h, x:x + w]
            # Dựa vào hình dạng đồ thị tìm những đỉnh cực đại
            hist = np.histogram(img_cnt[:, :, 2].ravel(), bins=256)
            for i in range(256):
                if hist[0].max() == hist[0][i]:
                    if i > 170:
                        if y > 10:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            # Trả về ảnh biển số xe sau khi đã contour
                            img_detect = img_detect[y - 10:10 + y + h, x - 10:10 + x + w]
                            return img_detect  # anh RGB
                        else:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            # Trả về ảnh biển số xe sau khi đã contour
                            img_detect = img_detect[y:10 + y + h, x - 10:10 + x + w]
                            return img_detect  # anh RGB

    # nếu số contour đếm dc == 0 thì xử lý lại
    if count_img == 0:
        contours, hierachy = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if y > 10:
                if 50 < w and 50 < h:
                    count_img += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # Trả về ảnh biển số xe sau khi đã contour
                    img_detect = img_detect[y - 10:10 + y + h, x - 10:10 + x + w]
                    return img_detect  # anh RGB
            else:
                if 50 < w and 50 < h:
                    count_img += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # Trả về ảnh biển số xe sau khi đã contour
                    img_detect = img_detect[y:10 + y + h, x - 10:10 + x + w]
                    return img_detect  # anh RGB
        if count_img == 0:
            return np.zeros((200, 300))


'''
Tìm vị trí của các số trên xe
'''


def location_num(im):
    im_cut = im.copy()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Loại bỏ noise
    gauss = cv2.GaussianBlur(im_gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Hàm xóa background
    morph_image = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(gauss, morph_image)
    ret, thresh_image = cv2.threshold(sub_morp_image, 120, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
        pts = np.int0(pts)
        if 40 < math.dist(pts[0], pts[1]) and 30 < math.dist(pts[1], pts[2]):  # Nếu như mà độ dài của các điểm trong bõ >30 thì lấy
            # print( math.dist(pts[0], pts[1]))
            # rint(math.dist(pts[1],pts[2]))
            for i in pts:
                box.append(i)
            cv2.drawContours(im , [pts], 0, (0, 0, 255), 2)
    up = []
    down = []
    # phân loại các điểm ở trên ở dưới để định vị chính xác biển
    for i in box:
        if i[1] < im.shape[0] / 2:
            up.append(i)
        else:
            down.append(i)
    for i in range(len(up) - 1):
        for j in range(i + 1, len(up)):
            if up[i][0] > up[j][0]:
                temp = up[i]
                up[i] = up[j]
                up[j] = temp
    for i in range(len(down) - 1):
        for j in range(i + 1, len(down)):
            if down[i][0] > down[j][0]:
                temp = down[i]
                down[i] = down[j]
                down[j] = temp
    point = []
    for i in up:
        point.append(i)
    for i in down:
        point.append(i)
    pts1 = np.float32(point)
    pts2 = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(im_cut, M, (300, 200))
    return dst  # Trả về ảnh đã được dectect


'''
Tìm kiếm vị trí của các kí tự trên biển
'''


def find_num(im):
    im_test = im.copy()
    im_cut = im.copy()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Xử dụng adaptive chuyển sang binary rõ các cạnh hơn
    im_bi = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 5)
    open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(im_bi, cv2.MORPH_OPEN, open)
    contours, hierachy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    up = []
    down = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(im_test, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # Đặt điều kiện cho việc contour để có thể nhận diện được đâu là số đâu không phải số
        if 0.25 < w / h < 0.65 and 55 < h < 100 and 20 < w and 1250 < w * h and 5 < x and x + w < 300:
            '''
             Chia các ảnh sau khi đã contour thành 2 phần
             Up là các số ở trên
             Down là các số ở dưới
            '''
            if y < round(im_cut.shape[0] / 3):
                if x > 200 or x < 150:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    up.append([x, y, w, h])
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
                down.append([x, y, w, h])    # Sắp xếp thứ tự của từng số
    for i in range(len(up) - 1):
        for j in range(i + 1, len(up)):
            if up[i][0] > up[j][0]:
                temp = up[i]
                up[i] = up[j]
                up[j] = temp
    for i in range(len(down) - 1):
        for j in range(i + 1, len(down)):
            if down[i][0] > down[j][0]:
                temp = down[i]
                down[i] = down[j]
                down[j] = temp
    # print(up)
    detect = up
    for i in range(len(down)):
        detect.append(down[i])
    # Lưu kí tự lại
    num = []
    for a in detect:
        numbers = im_cut[a[1] - 1: 1 + a[1] + a[3], a[0] - 1: 1 + a[0] + a[2]]
        num.append(numbers)
    # Trả về list kí tự đã được detect
    return num


'''
Nhận diện từng kí tự trong list(num)
'''


def detect(im):
    global num_list
    im = cv2.resize(im, (70, 100))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(im_gray, 11, 75, 75)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)
    thresh_image = ~thresh_image
    im_median = cv2.medianBlur(thresh_image, 13)
    im_gauss = cv2.GaussianBlur(im_median, (13, 13), 0)
    im_blur = cv2.blur(im_gauss, (13, 13))

    # Dùng hàm trung bình để chuyển từ 2D sang 1D
    list_x = []
    list_y = []
    i = 0
    while i < 99:
        ai = statistics.mean(im_blur[i])
        j = i + 1
        aj = statistics.mean(im_blur[j])
        while ai == aj and j < 99:
            j += 1
            aj = statistics.mean(im_blur[j])
        list_y.append(ai)
        list_x.append(i)
        i = j
    #plt.plot(list_x, list_y)
    #plt.show()

    # Lập danh sách tạo độ của đồ thị hình ảnh 1D
    # Tọa độ cực đại của hình
    max = []

    # Tọa độ cực tiểu của hình
    min = []

    # Tìm cực đại, cực tiểu của hình ảnh
    for i in range(1, len(list_x) - 1):
        if list_y[i - 1] < list_y[i] and list_y[i + 1] < list_y[i]:
            max.append([list_x[i], list_y[i]])
        elif list_y[i - 1] > list_y[i] and list_y[i + 1] > list_y[i]:
            min.append([list_x[i], list_y[i]])

    print(len(max), len(min))

    # Xác định dựa trên các đặc trưng của số
    count = 0;
    if len(max) == 1:
        #print(str(max[0][0]) + ':' + str(max[0][1]))
        if len(min) == 0:
            if 65< max[0][0] <= 75:
                if 200< max[0][1] < 255:
                    if count == 0:
                        count += 1
                        num_list.append("4")
                        print("Detect 4")
                    else:
                        print("Dectect sai với 4")
                        for i in range(count):
                            num_list.remove(num_list[-1])
                        num_list.append("?")
                # Detect 4
            if 5 < max[0][0] < 20:
                if 150 < max[0][1] < 200:
                    if count == 0:
                        count += 1
                        num_list.append("1")
                        print("Detect 1")
                    else:
                        print("Dectect sai với 1")
                        for i in range(count):
                            num_list.remove(num_list[-1])
                        num_list.append("?")
                # Detect 1
                if 200< max[0][1] < 250:
                    if count == 0:
                        count += 1
                        num_list.append("7")
                        print("Detect 7")
                    else:
                        print("Dectect sai với 7")
                        for i in range(count):
                            num_list.remove(num_list[-1])
                        num_list.append("?")
                # Detect 7

        if len(min) == 1:
            #print(str(min[0][0]) + ':' + str(min[0][1]))
            if 55 < max[0][0] < 90:
                if 100 < max[0][1]  < 200:
                    if 35 < min[0][0] < 80:
                        if 95 < min[0][1] < 110:
                            if count == 0:
                                count += 1
                                num_list.append("7")
                                print("Detect 7")
                            else:
                                print("Dectect sai với 7")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 7
                        if 85 < min[0][1] < 95:
                            if count == 0:
                                count += 1
                                num_list.append("1")
                                print("Detect 1")
                            else:
                                print("Dectect sai với 1")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 1
                    if 20 < min[0][0] < 45:
                        if 70 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("7")
                                print("Detect 7")
                            else:
                                print("Dectect sai với 7")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 7
                if 200 < max[0][1] < 250:
                    if 30 < min[0][0] < 50:
                        if 90 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("4")
                                print("Detect 4")
                            else:
                                print("Dectect sai với 4")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 4
            if 5 < max[0][0] < 20:
                if 150 < max[0][1] < 205:
                    if 10 < min[0][0] <=30:
                        if 80 <= min[0][1] < 90:
                            if count == 0:
                                count += 1
                                num_list.append("2")
                                print("Detect 2")
                            else:
                                print("Dectect sai với 2")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        if 70 < min[0][1] < 80:
                            if count == 0:
                                count += 1
                                num_list.append("7")
                                print("Detect 7")
                            else:
                                print("Dectect sai với 7")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                    if 30 < min[0][0] < 40:
                        if 95 < min[0][1] < 105:
                            if count == 0:
                                count += 1
                                num_list.append("1")
                                print("Detect 1")
                            else:
                                print("Dectect sai với 1")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 1
                        if 80 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("2")
                                print("Detect 2")
                            else:
                                print("Dectect sai với 2")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 2
                    if 40 < min[0][0] < 60:
                        if 80 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("2")
                                print("Detect 2")
                            else:
                                print("Dectect sai với 2")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 2
                    if 60 < min[0][0] < 70:
                        if 80 < min[0][1] < 90:
                            if count == 0:
                                count += 1
                                num_list.append("2")
                                print("Detect 2")
                            else:
                                print("Dectect sai với 2")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 2
                if 200 < max[0][1] < 250:
                    if 30 < min[0][0] < 40:
                        if 90 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("4")
                                print("Detect 4")
                            else:
                                print("Dectect sai với 4")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 4
                    if 10 < min[0][0] <= 30:
                        if 70 < min[0][1] < 100:
                            if count == 0:
                                count += 1
                                num_list.append("7")
                                print("Detect 7")
                            else:
                                print("Dectect sai với 7")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 7


    if len(max) == 2:
        """
        len(min) = 2
        """
        if len(min) == 2:
            #print(str(max[0][0]) + ':' + str(max[0][1]), str(max[1][0]) + ':' + str(max[1][1]))
            #print(str(min[0][0]) + ':' + str(min[0][1]), str(min[1][0]) + ':' + str(min[1][1]))
            if  max[1][0] < 80:
                if 20 < min[0][0] < 50:
                    if 60 < min[0][1] < 100:
                        if 50 < min[1][0] < 100:
                            if 100 < min[1][1] < 150:
                                if count == 0:
                                    count += 1
                                    num_list.append("6")
                                    print("Detect 6")
                                else:
                                    print("Dectect sai với 6")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 6
                            if 80 < min[1][1] < 100:
                                if count == 0:
                                    count += 1
                                    num_list.append("2")
                                    print("Detect 2")
                                else:
                                    print("Dectect sai với 2")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 2
                            if 60 < min[1][1] < 80:
                                if count == 0:
                                    count += 1
                                    num_list.append("5")
                                    print("Detect 5")
                                else:
                                    print("Dectect sai với 5")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 2

                    if 100 < min[0][1] < 120:
                        if 50 < min[1][0] < 70:
                            if 80 < min[1][1] < 110:
                                if count == 0:
                                    count += 1
                                    num_list.append("7")
                                    print("Detect 7")
                                else:
                                    print("Dectect sai với 7")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 7
                        if 70 <= min[1][0] < 95:
                            if 100 < min[1][1] < 120:
                                if count == 0:
                                    count += 1
                                    num_list.append("1")
                                    print("Detect 1")
                                else:
                                    print("Dectect sai với 1")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 1
                    if 120 <= min[0][1] < 150:
                        if 50 < min[1][1] < 80:
                            if count == 0:
                                count += 1
                                num_list.append("9")
                                print("Detect 9")
                            else:
                                print("Dectect sai với 9")
                                for i in range(count):
                                    num_list.remove(num_list[-1])
                                num_list.append("?")
                        # Detect 9
            elif max[1][0] >=80:
                if 20 < min[0][0] < 65:
                    if 60 < min[0][1] < 100:
                        if 50 < min[1][0] < 70:
                            if 30 < min[1][1] < 85:
                                if count == 0:
                                    count += 1
                                    num_list.append("5")
                                    print("Detect 5")
                                else:
                                    print("Dectect sai với 5")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 5
                        if 70 <= min[1][0] < 100:
                            if 80 <= min[1][1] < 250:
                                if count == 0:
                                    count += 1
                                    num_list.append("2")
                                    print("Detect 2")
                                else:
                                    print("Dectect sai với 2")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 2



        """
        len(min) = 1
        """
        if len(min) == 1:
            #print(str(max[0][0]) + ':' + str(max[0][1]), str(max[1][0]) + ':' + str(max[1][1]))
            #print(str(min[0][0]) + ':' + str(min[0][1]))
            if max[1][0] < 80:
                if 20 < min[0][0] < 80:
                    if 85 < min[0][1] < 130:
                        if max[0][1] > max[1][1]:
                            if max[0][1] < 220 and 0.2*max[0][1] < max[0][1] - max[1][1] < 0.55*max[0][1]:
                                    if count == 0:
                                        count += 1
                                        num_list.append("1")
                                        print("Detect 1")
                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                    else:
                                        print("Dectect sai với 1")
                                        for i in range(count):
                                            num_list.remove(num_list[-1])
                                        num_list.append("?")
                            # Detect 1

                            elif max[0][1] >= 220 and 0.4 * max[0][1] < max[0][1] - max[1][1] < 0.65 * max[0][1]:
                                if count == 0:
                                    count += 1
                                    num_list.append("7")
                                    print("Detect 7")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 7")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 7
                        elif max[0][1] < max[1][1]:
                            if 0.25 * max[1][1] < max[1][1] - max[0][1] < 0.65 * max[1][1]:
                                if count == 0:
                                    count += 1
                                    num_list.append("4")
                                    print("Detect 4")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 4")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 4
            if max[1][0] >= 80:
                if 20 < min[0][0] < 70:
                    if 75 < min[0][1] < 140:
                        if max[0][1] > max[1][1]:
                            if max[0][1] < 220 and 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.55 * max[0][1]:
                                    if count == 0:
                                        count += 1
                                        num_list.append("1")
                                        print("Detect 1")
                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                    else:
                                        print("Dectect sai với 1")
                                        for i in range(count):
                                            num_list.remove(num_list[-1])
                                        num_list.append("?")
                            # Detect 1

                            elif max[0][1] >= 220 and 0.4 * max[0][1] < max[0][1] - max[1][1] < 0.65 * max[0][1]:
                                if count == 0:
                                    count += 1
                                    num_list.append("7")
                                    print("Detect 7")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 7")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 7
                        if 0.2 * max[1][1] < max[1][1] - min[0][1] < 0.5 * max[1][1]:
                            if max[0][1] >= max[1][1] and max[0][1] - max[1][1] < 0.25 * max[0][1] :
                                if count == 0:
                                    count += 1
                                    num_list.append("0")
                                    print("Detect 0")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 0")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            if max[0][1] < max[1][1] and max[1][1] - max[0][1] < 0.25 * max[1][1] :
                                if count == 0:
                                    count += 1
                                    num_list.append("0")
                                    print("Detect 0")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 0")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 0
                        if 0.5 * max[1][1] < max[1][1] - min[0][1] < 0.9 * max[1][1]:
                                if count == 0:
                                    count += 1
                                    num_list.append("2")
                                    print("Detect 2")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 2")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 2
                    if 20 < min[0][1] <= 75:
                        if 0.5 * max[1][1] < max[1][1] - min[0][1] < 0.9 * max[1][1]:
                                if count == 0:
                                    count += 1
                                    num_list.append("2")
                                    print("Detect 2")
                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                else:
                                    print("Dectect sai với 2")
                                    for i in range(count):
                                        num_list.remove(num_list[-1])
                                    num_list.append("?")
                            # Detect 2










    if len(max) == 3:
        print(str(max[0][0]) + ':' + str(max[0][1]), str(max[1][0]) + ':' + str(max[1][1]),str(max[2][0]) + ':' + str(max[2][1]))
        print(str(min[0][0]) + ':' + str(min[0][1]), str(min[1][0]) + ':' + str(min[1][1]))
        if len(min) < 4:
            # max[1][0] < 45
            if max[1][0] < 45:
                if 70 <= max[2][0] <= 80:
                    if 20 < min[0][0] < 30:
                        if 58 < min[1][0] < 75:
                            if min[0][1] < 110:
                                if min[1][1] < 100:
                                    if 0.4 * max[0][1] < max[0][1] - min[0][1] < 0.8 * max[0][1]:
                                        if max[0][1] <= max[1][1] and max[1][1] - max[0][1] < 0.3 * max[1][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                        if max[1][1] < max[0][1] and max[0][1] - max[1][1] < 0.3 * max[0][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                            else:
                                                print("Dectect sai với 5")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                    # Detect 5


                    if 20 <= min[0][0] < 40:
                        if 40 < min[1][0] < 80:
                            if 95 < min[0][1] < 130:
                                if 95 < min[1][1] < 130:
                                    if max[1][1] < max[0][1] and max[2][1] < max[0][1]:
                                        if max[2][1] <= max[1][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        if max[1][1] < max[2][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                        # Detect 1
                                    #if max[2][1] > max[1][1] and max[2][1] > max[0][1]:
                                    #    if max[2][1] - max[1][1] > 0.45 * max[2][1] and max[2][1] - max[0][1] > 0.45 * max[2][1]:
                                    #        if count == 0:
                                    #            count += 1
                                    #            num_list.append("4")
                                    #            print("Detect 4")
                                    #            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                    #        else:
                                    #            print("Dectect sai với 4")
                                    #            for i in range(count):
                                    #                num_list.remove(num_list[-1])
                                    #    if max[0][1] > max[1][1] and max[0][1] - max[1][1] < 0.1 * max[0][1]:
                                    #       if count == 0:
                                    #            count += 1
                                    #            num_list.append("4")
                                    #            print("Detect 4")
                                    #            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                    #       else:
                                    #            print("Dectect sai với 4")
                                    #            for i in range(count):
                                    #                num_list.remove(num_list[-1])
                        # Detect 4

                        if 80 < min[0][1] <= 105:
                            if 60 < min[1][1] < 100:
                                if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                    if 0 < max[1][1] - min[0][1] <= 0.15 * max[1][1]:
                                        if 0.4 * max[2][1] < max[2][1] - max[1][1] < 0.7 * max[2][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("2")
                                                print("Detect 2")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 2")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                            # Detect 2


                            if 90 < min[1][1] < 110:
                                if max[0][1] > max[1][1] and max[0][1] > max[2][1]:
                                    if max[0][1] - max[1][1] > 0.45 * max[0][1] and max[0][1] - max[2][1] > 0.45 * max[0][1]:
                                        if max[1][1] >= max[2][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("7")
                                                print("Detect 7")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 7")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                                        if max[2][1] > max[1][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("7")
                                                print("Detect 7")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 7")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                        # Detect 7


                        if 105 <= min[0][1] < 120:
                            if max[2][1] > max[0][1] and max[2][1] > max[1][1]:
                                if max[2][1] - max[0][1] > 0.4 * max[2][1] and max[2][1] - max[1][1] > 0.4 * max[2][1]:
                                    if max[0][1] >= max[1][1] and max[0][1] - max[1][1] < 0.1 * max[0][1]:
                                        if max[1][1] > max[0][1] and max[1][1] - max[0][1] < 0.1 * max[1][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("4")
                                                print("Detect 4")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 4")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                        if max[0][1] > max[1][1] and max[0][1] - max[1][1] < 0.1 * max[0][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("4")
                                                print("Detect 4")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 4")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                # Detect 4


                if 80 < max[2][0]:
                    if 15 < min[0][0] < 45:
                        if 50 <= min[1][0] < 75:
                            if 45 < min[0][1] <= 90:
                                if 90 < min[1][1] < 145:
                                    if 0.15 * min[1][1] < min[1][1] - min[0][1] < min[1][1] * 0.7:
                                        if 0.45 * max[1][1] < max[1][1] - min[0][1] < max[1][1] * 0.85:
                                            if count == 0:
                                                count += 1
                                                num_list.append("6")
                                                print("Detect 6")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 6")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                            # Detect 6
                            if 45 < min[0][1] < 60:
                                if 70 < min[1][1] < 90:
                                    if 0.4 * max[0][1] < max[0][1] - min[0][1] < 0.8 * max[0][1]:
                                        if max[0][1] < max[1][1] and max[1][1] - max[0][1] < 0.3 * max[1][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                        if max[1][1] <= max[0][1] and max[0][1] - max[1][1] < 0.3 * max[0][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                            else:
                                                print("Dectect sai với 5")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                                 # Detect 5

                            if 65 < min[0][1] < 110:
                                if 60 < min[1][1] < 100 :
                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if 0.05 * max[1][1] < max[1][1] - min[1][1] < 0.6 * max[1][1]:
                                            if 0.2 * max[2][1] < max[2][1] - max[1][1] < 0.5 * max[2][1]:
                                                if 0.3 * max[0][1] < max[0][1] - max[1][1] < 0.4 * max[0][1]:
                                                    if 0 < max[0][1] - max[2][1] < 0.4 * max[0][1]:
                                                        if count == 0:
                                                            count += 1
                                                            num_list.append("3")
                                                            print("Detect 3")
                                                        # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                        else:
                                                            print("Dectect sai với 3")
                                                            for i in range(count):
                                                                num_list.remove(num_list[-1])
                                                            num_list.append("?")
                                # Detect 3

                                    if 0.30 * max[0][1] < max[0][1] - min[0][1] < 0.8 * max[0][1]:
                                        if max[0][1] < max[1][1] and max[1][1] - max[0][1] < 0.3 * max[1][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                        if max[1][1] <= max[0][1] and max[0][1] - max[1][1] < 0.3 * max[0][1]:
                                            # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                            if count == 0:
                                                count += 1
                                                num_list.append("5")
                                                print("Detect 5")
                                            else:
                                                print("Dectect sai với 5")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                                 # Detect 5
                                if 90 < min[1][1] < 110:
                                    if max[0][1] > max[1][1] and max[0][1] > max[2][1]:
                                        if max[0][1] - max[1][1] > 0.45 * max[0][1] and max[0][1] - max[2][1] > 0.45 * max[0][1]:
                                            if max[1][1] >= max[2][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                            if max[2][1] > max[1][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                        # Detect 7


                        if 40 < min[1][0] <= 80:
                            if 106 < min[0][1] < 160:
                                if 90 < min[1][1] < 145:
                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                            if 0.1 * max[2][1] < max[2][1] - max[1][1] < 0.35 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("0")
                                                    print("Detect 0")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 0")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                            # Detect 0


                            if 100 < min[0][1] < 130:
                                if 100 < min[1][1] < 130:
                                    if max[1][1] < max[0][1] and max[2][1] <= max[0][1]:
                                        if max[2][1] <= max[1][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        if max[1][1] < max[2][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                            # Detect 1


                            if 80 < min[0][1] < 105:
                                if 60 < min[1][1] < 100:
                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if 0 < max[1][1] - min[0][1] <= 0.15 * max[1][1]:
                                            if 0.4 * max[2][1] < max[2][1] - max[1][1] < 0.7 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("2")
                                                    print("Detect 2")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 2")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
            # Detect 2


            # max[1][0] > 45
            elif 45 <= max[1][0]:
                if 80 < max[2][0]:
                    if 20 < min[0][0] <= 50:
                        if 50 < min[1][0] < 90:
                            if 35 < min[0][1] < 85:
                                if 100 < min[1][1] < 170:
                                    if 0.15 * min[1][1] < min[1][1] - min[0][1] < min[1][1] * 0.8:
                                        if 0.50 * max[1][1] < max[1][1] - min[0][1] < max[1][1] * 0.85:
                                            if count == 0:
                                                count += 1
                                                num_list.append("6")
                                                print("Detect 6")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 6")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                                    # Detect 6

                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if 0.15 * max[1][1] < max[1][1] - min[0][1] < 0.6 * max[1][1]:
                                            if 0.2 * max[2][1] < max[2][1] - max[1][1] < 0.5 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("3")
                                                    print("Detect 3")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 3")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                            # Detect 3
                            if 85 <= min[0][1] < 110:
                                if 90 < min[1][1] < 110:
                                    if max[0][1] > max[1][1] and max[0][1] > max[2][1]:
                                        if max[0][1] - max[1][1] > 0.45 * max[0][1] and max[0][1] - max[2][1] > 0.45 * max[0][1]:
                                            if max[1][1] >= max[2][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                            if max[2][1] > max[1][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                            # Detect 7



                            if 90 < min[0][1] < 165:
                                if 20 < min[1][1] < 90:
                                    if 0.25 * min[0][1] < min[0][1] - min[1][1] < min[0][1] * 0.85:
                                        if 0.40 * max[1][1] < max[1][1] - min[1][1] < max[1][1] * 0.9:
                                            if count == 0:
                                                count += 1
                                                num_list.append("9")
                                                print("Detect 9")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 9")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                                # Detect 9


                                if 90 < min[1][1] < 145:
                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if  max[2][1] - max[1][1] < 0.4 * max[2][1]:
                                            if 0 < max[1][1] - min[0][1] < 0.2 * max[1][1]:
                                                if max[0][1] > max[2][1] and max[0][1] - max[2][1] < 0.2*max[0][1]:
                                                    if count == 0:
                                                        count += 1
                                                        num_list.append("0")
                                                        print("Detect 0")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                    else:
                                                        print("Dectect sai với 0")
                                                        for i in range(count):
                                                            num_list.remove(num_list[-1])
                                                        num_list.append("?")
                                                if max[0][1] < max[2][1] and max[2][1] - max[0][1] <0.2*max[2][1]:
                                                    if count == 0:
                                                        count += 1
                                                        num_list.append("0")
                                                        print("Detect 0")
                                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                    else:
                                                        print("Dectect sai với 0")
                                                        for i in range(count):
                                                            num_list.remove(num_list[-1])
                                                        num_list.append("?")
                                    # Detect 0


                                    if max[1][1] < max[0][1] and max[2][1] <= max[0][1]:
                                        if max[2][1] <= max[1][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        #Detect 1
                                            if max[0][1] - max[1][1] < 0.2*max[0][1]:
                                                if max[0][1] > max[2][1] and max[0][1] - max[2][1] < 0.2*max[0][1]:
                                                    if count == 0:
                                                        count += 1
                                                        num_list.append("0")
                                                        print("Detect 0")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                    else:
                                                        print("Dectect sai với 0")
                                                        for i in range(count):
                                                            num_list.remove(num_list[-1])
                                                        num_list.append("?")
                                                #Detect 0
                                                if max[0][1] < max[2][1] and max[2][1] - max[0][1] <0.2*max[2][1]:
                                                    if count == 0:
                                                        count += 1
                                                        num_list.append("0")
                                                        print("Detect 0")
                                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                    else:
                                                        print("Dectect sai với 0")
                                                        for i in range(count):
                                                            num_list.remove(num_list[-1])
                                                        num_list.append("?")
                                            #Detect 0

                                        if max[1][1] < max[2][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                            if 0.2 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                # Detect 1                                                num_list.append("?")

                                if 90 < min[1][1] < 140:
                                    if max[1][1] >= max[2][1]:
                                        if 0.05 * max[1][1] < max[1][1] - max[2][1] < 0.3 * max[1][1]:
                                            if count == 0:
                                                count += 1
                                                num_list.append("8")
                                                print("Detect 8")
                                            # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                            else:
                                                print("Dectect sai với 8")
                                                for i in range(count):
                                                    num_list.remove(num_list[-1])
                                                num_list.append("?")
                            # Detect 8

                            if 70 < min[0][1] < 105:
                                if 60 < min[1][1] < 100:
                                    if max[1][1] < max[0][1] and max[1][1] < max[2][1]:
                                        if 0.15 * max[1][1] < max[1][1] - min[0][1] < 0.5 * max[1][1]:
                                            if 0.15 * max[2][1] < max[2][1] - max[1][1] < 0.5 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("3")
                                                    print("Detect 3")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 3")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        # Detect 3

                                        if 0 < max[1][1] - min[0][1] < 0.2 * max[1][1]:
                                            if 0.4 * max[2][1] < max[2][1] - max[1][1] < 0.7 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("2")
                                                    print("Detect 2")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 2")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                    # Detect 2

                                    if max[0][1] >= max[1][1] >= max[2][1]:
                                        if min[0][1] < 110:
                                            if min[1][1] < 100:
                                                if 0.4 * max[0][1] < max[0][1] - min[0][1] < 0.8 * max[0][1]:
                                                    if max[0][1] <= max[1][1] and max[1][1] - max[0][1] < 0.3 * max[1][1]:
                                                        # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                                        if count == 0:
                                                            count += 1
                                                            num_list.append("5")
                                                            print("Detect 5")
                                                        else:
                                                            print("Dectect sai với 5")
                                                            for i in range(count):
                                                                num_list.remove(num_list[-1])
                                                            num_list.append("?")
                                                    if max[1][1] < max[0][1] and max[0][1] - max[1][1] < 0.3 * max[0][1]:
                                                        # Nếu count == 0 nghĩa là detect không bị trùng lặp thì thực hiện
                                                        if count == 0:
                                                            count += 1
                                                            num_list.append("5")
                                                            print("Detect 5")
                                                        else:
                                                            print("Dectect sai với 5")
                                                            for i in range(count):
                                                                num_list.remove(num_list[-1])
                                                            num_list.append("?")

                if max[2][0] <= 80:
                    if 25 < min[0][0] < 50:
                        if 50 < min[1][0] < 80:
                            if 90 < min[0][1] < 130:
                                if 95 < min[1][1] < 130:
                                    if max[1][1] < max[0][1] and max[2][1] <= max[0][1]:
                                        if max[0][1]  < 230 and max[2][1] <= max[1][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                            if 0.1 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        if max[0][1] < 230 and max[1][1] < max[2][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                            if 0.1 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                    # Detect 1



                                        if max[0][1] > 230 and max[0][1] - max[1][1] > 0.45 * max[0][1] and max[0][1] - max[2][1] > 0.45 * max[0][1]:
                                            if max[1][1] >= max[2][1] and max[1][1] - max[2][1] < 0.1 * max[1][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                            if max[2][1] > max[1][1] and max[2][1] - max[1][1] < 0.1 * max[2][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("7")
                                                    print("Detect 7")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 7")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                        # Detect 7
                                    if max[2][1] > max[1][1] and max[2][1] > max[0][1]:
                                        if max[2][1] - max[1][1] > 0.45 * max[2][1] and max[2][1] - max[0][1] > 0.45 * max[2][1]:
                                            if max[1][1] > max[0][1] and max[1][1] - max[0][1] < 0.1 * max[1][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("4")
                                                    print("Detect 4")
                                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 4")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                            if max[0][1] > max[1][1] and max[0][1] - max[1][1] < 0.1 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("4")
                                                    print("Detect 4")
                                                    # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 4")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                    # Detect 4
                    if 50 <= min[0][0] < 60:
                        if 50 < min[1][0] < 80:
                            if 90 < min[0][1] < 130:
                                if 95 < min[1][1] < 130:
                                    if max[1][1] < max[0][1] and max[2][1] <= max[0][1]:
                                        if max[0][1] < 230 and max[2][1] <= max[1][1] and max[1][1] - max[2][1] < 0.1 * \
                                                max[1][1]:
                                            if 0.1 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                                        if max[0][1] < 230 and max[1][1] < max[2][1] and max[2][1] - max[1][1] < 0.1 * \
                                                max[2][1]:
                                            if 0.1 * max[0][1] < max[0][1] - max[1][1] < 0.5 * max[0][1]:
                                                if count == 0:
                                                    count += 1
                                                    num_list.append("1")
                                                    print("Detect 1")
                                                # Nếu count != 0 nghĩa là có 1 số vị trí detect bị trùng lặp => thay số bị trùng = "?"
                                                else:
                                                    print("Dectect sai với 1")
                                                    for i in range(count):
                                                        num_list.remove(num_list[-1])
                                                    num_list.append("?")
                    # Detect 1


for i in range(0, 300):
    start_time = time.time()
    image = image_input(i)
    # resize anh de xoa bot nhung vat the khong mong muon
    img = image[:280, 60:420]
    # chia anh thanh RGB de loai bo mau do cua den
    blue, green, red = cv2.split(img)
    img_pro = image_process(red, img)
    num_list = []
    print(i)
    if img_pro.max() != 0:
        img_detect = location_num(img_pro)
        number = find_num(img_detect)
        if len(number) != 1:
            for j in range(len(number)):
                if number[j].shape[0] != 0:
                    detect(number[j])
                    print('--------------------------------------')
        else:
            print("Ảnh lỗi")
    else:
        print("Ảnh lỗi")
    print(num_list)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    l=0
    for b in num_list:
        cv2.putText(image,b,(3*l+10,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        l=l+10
    cv2.imshow('img', image)
    cv2.waitKey(0)