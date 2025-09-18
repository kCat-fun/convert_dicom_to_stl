// JavaScript Document

/* 初期設定 */
document.cookie = 1;
var imgcnt = 1;
var imagfilename = new Array();
imagfilename[0] = '00000001.JPG'
imagfilename[1] = '00000002.JPG'
imagfilename[2] = '00000003.JPG'
imagfilename[3] = '00000004.JPG'
imagfilename[4] = '00000005.JPG'
imagfilename[5] = '00000006.JPG'
imagfilename[6] = '00000007.JPG'
imagfilename[7] = '00000008.JPG'
imagfilename[8] = '00000009.JPG'
imagfilename[9] = '00000010.JPG'
imagfilename[10] = '00000011.JPG'
imagfilename[11] = '00000012.JPG'
imagfilename[12] = '00000013.JPG'
imagfilename[13] = '00000014.JPG'
imagfilename[14] = '00000015.JPG'
imagfilename[15] = '00000016.JPG'
imagfilename[16] = '00000017.JPG'
imagfilename[17] = '00000018.JPG'
imagfilename[18] = '00000019.JPG'
imagfilename[19] = '00000020.JPG'
imagfilename[20] = '00000021.JPG'
imagfilename[21] = '00000022.JPG'
imagfilename[22] = '00000023.JPG'
imagfilename[23] = '00000024.JPG'
imagfilename[24] = '00000025.JPG'
imagfilename[25] = '00000026.JPG'
imagfilename[26] = '00000027.JPG'
imagfilename[27] = '00000028.JPG'
imagfilename[28] = '00000029.JPG'
imagfilename[29] = '00000030.JPG'
imagfilename[30] = '00000031.JPG'
imagfilename[31] = '00000032.JPG'
imagfilename[32] = '00000033.JPG'
imagfilename[33] = '00000034.JPG'
imagfilename[34] = '00000035.JPG'
imagfilename[35] = '00000036.JPG'
imagfilename[36] = '00000037.JPG'
imagfilename[37] = '00000038.JPG'
imagfilename[38] = '00000039.JPG'
imagfilename[39] = '00000040.JPG'
imagfilename[40] = '00000041.JPG'
imagfilename[41] = '00000042.JPG'
imagfilename[42] = '00000043.JPG'
imagfilename[43] = '00000044.JPG'
imagfilename[44] = '00000045.JPG'
imagfilename[45] = '00000046.JPG'
imagfilename[46] = '00000047.JPG'
imagfilename[47] = '00000048.JPG'
imagfilename[48] = '00000049.JPG'
imagfilename[49] = '00000050.JPG'
imagfilename[50] = '00000051.JPG'
imagfilename[51] = '00000052.JPG'
imagfilename[52] = '00000053.JPG'
imagfilename[53] = '00000054.JPG'
imagfilename[54] = '00000055.JPG'
imagfilename[55] = '00000056.JPG'
imagfilename[56] = '00000057.JPG'
imagfilename[57] = '00000058.JPG'
imagfilename[58] = '00000059.JPG'
imagfilename[59] = '00000060.JPG'
imagfilename[60] = '00000061.JPG'
imagfilename[61] = '00000062.JPG'
imagfilename[62] = '00000063.JPG'
imagfilename[63] = '00000064.JPG'
imagfilename[64] = '00000065.JPG'
imagfilename[65] = '00000066.JPG'
imagfilename[66] = '00000067.JPG'
imagfilename[67] = '00000068.JPG'
imagfilename[68] = '00000069.JPG'
imagfilename[69] = '00000070.JPG'
imagfilename[70] = '00000071.JPG'
imagfilename[71] = '00000072.JPG'
imagfilename[72] = '00000073.JPG'
imagfilename[73] = '00000074.JPG'
imagfilename[74] = '00000075.JPG'
imagfilename[75] = '00000076.JPG'
imagfilename[76] = '00000077.JPG'
imagfilename[77] = '00000078.JPG'
imagfilename[78] = '00000079.JPG'
imagfilename[79] = '00000080.JPG'
imagfilename[80] = '00000081.JPG'
imagfilename[81] = '00000082.JPG'
imagfilename[82] = '00000083.JPG'
imagfilename[83] = '00000084.JPG'
imagfilename[84] = '00000085.JPG'
imagfilename[85] = '00000086.JPG'
imagfilename[86] = '00000087.JPG'
imagfilename[87] = '00000088.JPG'
imagfilename[88] = '00000089.JPG'
imagfilename[89] = '00000090.JPG'
imagfilename[90] = '00000091.JPG'
imagfilename[91] = '00000092.JPG'
imagfilename[92] = '00000093.JPG'
imagfilename[93] = '00000094.JPG'
imagfilename[94] = '00000095.JPG'
imagfilename[95] = '00000096.JPG'
imagfilename[96] = '00000097.JPG'
imagfilename[97] = '00000098.JPG'
imagfilename[98] = '00000099.JPG'
imagfilename[99] = '00000100.JPG'
imagfilename[100] = '00000101.JPG'
imagfilename[101] = '00000102.JPG'
imagfilename[102] = '00000103.JPG'
imagfilename[103] = '00000104.JPG'
imagfilename[104] = '00000105.JPG'
imagfilename[105] = '00000106.JPG'
imagfilename[106] = '00000107.JPG'
imagfilename[107] = '00000108.JPG'
imagfilename[108] = '00000109.JPG'
imagfilename[109] = '00000110.JPG'
imagfilename[110] = '00000111.JPG'
imagfilename[111] = '00000112.JPG'
imagfilename[112] = '00000113.JPG'
imagfilename[113] = '00000114.JPG'
imagfilename[114] = '00000115.JPG'
imagfilename[115] = '00000116.JPG'
imagfilename[116] = '00000117.JPG'
imagfilename[117] = '00000118.JPG'
imagfilename[118] = '00000119.JPG'
imagfilename[119] = '00000120.JPG'

var imgmax = imagfilename.length,
    arr_img,
    display_img = document.getElementById('display_img'),
    pageno = document.getElementById('pageno');

function prepareImgView() {
	// 画像レンダリング
	for (var i = 0; i < imgmax; i++) {
		var _elem_img = document.createElement("img");
		_elem_img.src = imagfilename[i];
		_elem_img.setAttribute('id', 'img' + (i + 1));
		_elem_img.setAttribute('width', '512');
		_elem_img.setAttribute('height', '512');
		_elem_img.setAttribute('alt', '診断画像');
		_elem_img.style.verticalAlign = "bottom";
		if ( i === 0) {
			_elem_img.setAttribute('class', 'current');
		} else {
			_elem_img.style.display = "none";
		}
		display_img.appendChild(_elem_img);
	}
	arr_img = display_img.getElementsByTagName('img');
	// 画像番号セット
	feedPageNum(1);
}
addLoadEvent(prepareImgView);
