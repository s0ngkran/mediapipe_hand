
def get_thai(str_num):
	map = {
		# static sign
		'000': 'ก',
		'001': 'ต',
		'002': 'ส',
		'003': 'พ',
		'004': 'ห',
		'005': 'บ',
		'006': 'ร',
		'007': 'ว',
		'008': 'ด',
		'009': 'ฟ',
		'010': 'ล',
		'011': 'ย',
		'012': 'ม',
		'013': 'น',
		'014': 'อ',
		'015': 'อิ',
		'016': 'โอ',
		'017': 'ไอ',
		'018': 'ใอ',
	}
	key = str(str_num).zfill(3)
	return map[key]

def test():
	assert get_thai('000') == 'ก'
	assert get_thai(000) == 'ก'

def get_list_folder(path):
	import os
	for _,list_folder, __ in os.walk(path):
		break
	return list_folder

def get_list_img(path):
	import os
	for _,__,list_img in os.walk(path):
		break
	return list_img



if __name__ == '__main__':
	test()
	print('success')

