from matplotlib import pyplot as plt
import numpy as np
import collections

from skimage import io
from skimage.feature import match_template


img_a = io.imread('img/img_a.PNG', as_gray=True)
img_b = io.imread('img/img_b.PNG', as_gray=True)
img_b2 = io.imread('img/img_b2.PNG', as_gray=True)
img_c = io.imread('img/img_c.PNG', as_gray=True)
img_d = io.imread('img/img_d.PNG', as_gray=True)
img_e = io.imread('img/img_e.PNG', as_gray=True)
img_f = io.imread('img/img_f.PNG', as_gray=True)
img_g = io.imread('img/img_g.PNG', as_gray=True)
img_h = io.imread('img/img_h.PNG', as_gray=True)
img_h2 = io.imread('img/img_h2.PNG', as_gray=True)
img_i = io.imread('img/img_i.PNG', as_gray=True)
img_j = io.imread('img/img_j.PNG', as_gray=True)
img_k = io.imread('img/img_k.PNG', as_gray=True)
img_l = io.imread('img/img_l.PNG', as_gray=True)
img_m = io.imread('img/img_m.PNG', as_gray=True)
img_n = io.imread('img/img_n.PNG', as_gray=True)
img_n2 = io.imread('img/img_n2.PNG', as_gray=True)
img_o = io.imread('img/img_o.PNG', as_gray=True)
img_p = io.imread('img/img_p.PNG', as_gray=True)
img_q = io.imread('img/img_q.PNG', as_gray=True)
img_r = io.imread('img/img_r.PNG', as_gray=True)
img_s = io.imread('img/img_s.PNG', as_gray=True)
img_t = io.imread('img/img_t.PNG', as_gray=True)
img_t2 = io.imread('img/img_t2.PNG', as_gray=True)
img_u = io.imread('img/img_u.PNG', as_gray=True)
img_v = io.imread('img/img_v.PNG', as_gray=True)
img_w = io.imread('img/img_w.PNG', as_gray=True)
img_x = io.imread('img/img_x.PNG', as_gray=True)
img_y = io.imread('img/img_y.PNG', as_gray=True)
img_z = io.imread('img/img_z.PNG', as_gray=True)


sentence1 = io.imread('img/sentence1.PNG', as_gray=True)
sentence2 = io.imread('img/sentence2.PNG', as_gray=True)
sentence3 = io.imread('img/sentence3.PNG', as_gray=True)

#tablica obrazow
hieroglyphics = {
	'a': img_a,
	'b': [img_b, img_b2],
	'c': img_c,
	'd': img_d,
	'e': img_e,
	'f': img_f,
	'g': img_g,
	'h': [img_h, img_h2],
	'i': img_i,
	'j': img_j,
	'k': img_k,
	'l': img_l,
	'm': img_m,
	'n': [img_n, img_n2],
	'o': img_o,
	'p': img_p,
	'q': img_q,
	'r': img_r,
	's': img_s,
	't': [img_t, img_t2],
	'u': img_u,
	'v': img_v,
	'w': img_w,
	'x': img_x,
	'y': img_y,
	'z': img_z
}



#funkcja usuwajaca znalezione elementy
def clear_match(target, y, x):
	for i in range(15):
		for j in range(15):
			target[y+i][x] = 0
			target[y+i][x-j] = 0
			target[y+i][x+j] = 0

			target[y-i][x] = 0
			target[y-i][x-j] = 0
			target[y-i][x+j] = 0

	return target


#kod prezentujacy jak dziala clear_match
result = match_template(sentence2, img_d)
plt.imshow(result)
plt.show()
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
result = clear_match(result, y, x)
plt.imshow(result)
plt.show()
found = np.where(result>0.974)
print('znaleziono: ', found)



#funkcja znajdujaca znaki w obrazie, zwraca znalezione i przetlumaczone litery
def read_sentence(sentence, threshold):
	letter_and_position = {}
	for key in hieroglyphics:
		if isinstance(hieroglyphics[key], list):
			for h in range(len(hieroglyphics[key])):
				result = match_template(sentence, hieroglyphics[key][h])
				ij = np.unravel_index(np.argmax(result), result.shape)
				x, y = ij[::-1]

				while result[y][x] > threshold:
					letter_and_position[x] = key
					result = clear_match(result, y, x)
					ij = np.unravel_index(np.argmax(result), result.shape)
					x, y = ij[::-1]
		else:
			result = match_template(sentence, hieroglyphics[key])
			ij = np.unravel_index(np.argmax(result), result.shape)
			x, y = ij[::-1]

			while result[y][x] > threshold:
				letter_and_position[x] = key
				result = clear_match(result, y, x)
				ij = np.unravel_index(np.argmax(result), result.shape)
				x, y = ij[::-1]

	sorted_l = collections.OrderedDict(sorted(letter_and_position.items()))
	sorted_letters = sorted_l.values()
	return sorted_letters



s1 = read_sentence(sentence1, 0.92)
print(s1)

s2 = read_sentence(sentence2, 0.92)
print(s2)

s3 = read_sentence(sentence3, 0.92)
print(s3)
