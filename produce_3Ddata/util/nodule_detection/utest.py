import unittest
from test_nodule_detect import get_region_candidate_center


class MyTestCase(unittest.TestCase):
    def test_simage_input(self):
        input_path = '/home/fc/fc/database/3a091ba3-6c3e-48c4-912f-22256ae870b1.i'
        mask_path = '/home/fc/fc/test/lungmask'
        output_path = '/home/fc/fc/nodule_detect'

        from sccore.src.etl.etl import ETL
        etl = ETL()
        simage = etl.read_instance(input_path)
        import pickle

        f = open('lungmask', 'rb')
        mask = pickle.load(f)
        f.close()
        mask[mask > 0] = 1
        nodule_mask = get_region_candidate_center(simage, mask)


if __name__ == '__main__':
    unittest.main()
