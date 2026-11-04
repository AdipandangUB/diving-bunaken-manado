"""Data georeferensi titik penyelaman & perjumpaan penyu, TN Bunaken.

SUMBER : dua peta "Peta Lokasi Perjumpaan Penyu" (Pulau Bunaken 1:50.000 dan Pulau Siladen 1:20.000).
METODE : posisi piksel titik/ikon dideteksi dari gambar (warna + template matching), lalu dikonversi ke
         WGS84 memakai garis grid lintang-bujur pada peta (regresi linear). Dua titik penyu di tepi timur
         peta Bunaken = titik yang sama dengan peta Siladen (selisih < 15 m) -> dipakai versi Siladen.
AKURASI: sekitar +/-15 m (Siladen) dan +/-40 m (Bunaken); ikon di peta digambar di dekat, bukan tepat di
         atas, titik selam sebenarnya. Verifikasi dengan GPS/BTNB sebelum dipakai operasional.
"""

# (nama, bujur, lintang, pulau, kuota penyelam bersamaan)
DIVE_SPOTS = [
    ("Mike's Point", 124.7341, 1.63554, 'Bunaken', 12),
    ("Matchikos's Point", 124.77247, 1.63336, 'Bunaken', 12),
    ("Raymond's Point", 124.73077, 1.6259, 'Bunaken', 12),
    ('Bunaken Timur I', 124.78606, 1.62174, 'Bunaken', 12),
    ('Tengah', 124.72824, 1.61969, 'Bunaken', 12),
    ('Tawara', 124.75768, 1.61672, 'Bunaken', 12),
    ('Alban Point', 124.74889, 1.61467, 'Bunaken', 12),
    ('Bunaken Timur II', 124.78726, 1.61216, 'Bunaken', 12),
    ('Fukui I', 124.74196, 1.61177, 'Bunaken', 15),
    ('Mandolin', 124.72744, 1.61137, 'Bunaken', 20),
    ('Fukui II', 124.74276, 1.60761, 'Bunaken', 15),
    ("Ron's Point", 124.73943, 1.60345, 'Bunaken', 12),
    ('Cela-Cela', 124.76248, 1.60219, 'Bunaken', 12),
    ('Pangalisan', 124.78726, 1.60133, 'Bunaken', 12),
    ('Likuan III', 124.76368, 1.59816, 'Bunaken', 20),
    ('Likuan II', 124.76501, 1.59599, 'Bunaken', 20),
    ('Likuan I', 124.76954, 1.59222, 'Bunaken', 20),
    ('Muka Kampung I', 124.77407, 1.59176, 'Bunaken', 15),
    ('Muka Kampung II', 124.7794, 1.59136, 'Bunaken', 15),
    ('Siladen Utara', 124.79775, 1.64181, 'Siladen', 18),
    ('Siladen Timur', 124.80959, 1.63473, 'Siladen', 18),
    ('Siladen', 124.80927, 1.62667, 'Siladen', 18),
]

# (jenis, bujur, lintang, pulau) ; jenis: hijau | sisik | lekang
TURTLES = [
    ('hijau', 124.73186, 1.61545, 'Bunaken'),
    ('hijau', 124.73288, 1.61131, 'Bunaken'),
    ('hijau', 124.73295, 1.62134, 'Bunaken'),
    ('hijau', 124.7337, 1.62394, 'Bunaken'),
    ('hijau', 124.73723, 1.63207, 'Bunaken'),
    ('hijau', 124.73802, 1.60823, 'Bunaken'),
    ('hijau', 124.74505, 1.61708, 'Bunaken'),
    ('hijau', 124.75285, 1.62955, 'Bunaken'),
    ('hijau', 124.75586, 1.62835, 'Bunaken'),
    ('hijau', 124.75779, 1.61848, 'Bunaken'),
    ('hijau', 124.76052, 1.61832, 'Bunaken'),
    ('hijau', 124.76487, 1.61714, 'Bunaken'),
    ('hijau', 124.76598, 1.60315, 'Bunaken'),
    ('hijau', 124.76682, 1.59844, 'Bunaken'),
    ('hijau', 124.76807, 1.61165, 'Bunaken'),
    ('hijau', 124.76902, 1.60654, 'Bunaken'),
    ('hijau', 124.76966, 1.62933, 'Bunaken'),
    ('hijau', 124.77127, 1.59538, 'Bunaken'),
    ('hijau', 124.7772, 1.62761, 'Bunaken'),
    ('hijau', 124.78241, 1.60681, 'Bunaken'),
    ('hijau', 124.78259, 1.60228, 'Bunaken'),
    ('hijau', 124.78281, 1.61562, 'Bunaken'),
    ('hijau', 124.78326, 1.59901, 'Bunaken'),
    ('hijau', 124.77674, 1.59414, 'Bunaken'),
    ('hijau', 124.7812, 1.59407, 'Bunaken'),
    ('hijau', 124.73237, 1.61315, 'Bunaken'),
    ('hijau', 124.76668, 1.60371, 'Bunaken'),
    ('hijau', 124.76708, 1.59915, 'Bunaken'),
    ('hijau', 124.76848, 1.59605, 'Bunaken'),
    ('sisik', 124.73755, 1.63281, 'Bunaken'),
    ('sisik', 124.75589, 1.61755, 'Bunaken'),
    ('sisik', 124.76849, 1.59634, 'Bunaken'),
    ('sisik', 124.73217, 1.61606, 'Bunaken'),
    ('sisik', 124.76048, 1.61916, 'Bunaken'),
    ('sisik', 124.76708, 1.59948, 'Bunaken'),
    ('sisik', 124.78247, 1.60747, 'Bunaken'),
    ('sisik', 124.75069, 1.61639, 'Bunaken'),
    ('sisik', 124.76668, 1.60384, 'Bunaken'),
    ('lekang', 124.73225, 1.61793, 'Bunaken'),
    ('lekang', 124.73941, 1.6134, 'Bunaken'),
    ('lekang', 124.73537, 1.62834, 'Bunaken'),
    ('hijau', 124.80139, 1.63925, 'Siladen'),
    ('hijau', 124.80706, 1.63274, 'Siladen'),
    ('hijau', 124.79592, 1.63143, 'Siladen'),
    ('hijau', 124.80725, 1.62973, 'Siladen'),
    ('hijau', 124.79691, 1.62957, 'Siladen'),
    ('sisik', 124.79885, 1.63902, 'Siladen'),
]

SPECIES = {
    "hijau": dict(nama="Penyu Hijau", latin="Chelonia mydas", rgb=(46, 204, 64), poin=10,
                  fakta="Penyu dewasa memakan lamun & alga sehingga padang lamun tetap sehat."),
    "sisik": dict(nama="Penyu Sisik", latin="Eretmochelys imbricata", rgb=(231, 76, 60), poin=25,
                  fakta="Pemakan spons terumbu karang; berstatus Kritis (CR) menurut IUCN."),
    "lekang": dict(nama="Penyu Lekang", latin="Lepidochelys olivacea", rgb=(241, 196, 15), poin=15,
                   fakta="Salah satu penyu laut terkecil; memakan krustasea, ubur-ubur, dan moluska."),
}
