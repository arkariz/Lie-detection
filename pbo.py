# #object
# class mobil:
#     #atribut
#     def __init__(self, nama, warna):
#         self.nama = nama
#         self.warna = warna
#
# def majuKedepan():
#   tulis kode
#
# mobil_biru = mobil('mobil warna biru', 'biru')
# mobil_merah = mobil('mobil warna merah', 'merah')

#
# class LuasPersegi:
#     def __init__(self, sisi):
#         self.sisi = sisi
#         self.luas = sisi * sisi
#
#
# print(LuasPersegi(1000).luas)
# print(LuasPersegi(12).luas)
# print(LuasPersegi(15).luas)
#
# sisipersegi1 = 10
# luaspersegi1 = sisipersegi1 * sisipersegi1
# print(luaspersegi1)
#
# sisipersegi2 = 100
# luaspersegi2 = sisipersegi2 * sisipersegi2
# print(luaspersegi2)


# blueprint object
class Color:
    # attribute
    def _init_(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
        self.menghitungKodeWarna()

    # method
    def menghitungKodeWarna(self):
        self.value = '#{:02x}{:02x}{:02x}'.format(self.r, self.g, self.b)


# instansiasi object
yellow = Color.menghitungKodeWarna(255, 255, 0)
print(yellow.value)
#
# #instansiasi object
# blue = Color(0, 0, 255)
# print(blue.menghitungKodeWarna())

# Ubah program tersebut menjadi program yang menggunakan cara yang kedua
# (dengan metode) dan cara yang ketiga (dengan campuran atribut/variabel dan metode).
