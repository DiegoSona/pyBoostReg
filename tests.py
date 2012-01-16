
import unittest
from gen_redundant_data import genRedundantData

class GenRedundantData(unittest.TestCase):
	
	def testGnRedundantData(self):
		Y,X = genRedundantData(100,6, 2, 2)
		# Columns 2 and 3 are equal to column 0 and columns 4 and 5 equal to 1
		# X1 = Y - X0
		self.assertEqual(X[:,0].all(), X[:,2].all())
		self.assertEqual(X[:,0].all(), X[:,3].all())
		self.assertEqual(X[:,1].all(), X[:,4].all())
		self.assertEqual(X[:,1].all(), X[:,5].all())
		self.assertEqual((Y - X[:,0]).all(), X[:,1].all())
		
	
		
		
if __name__ == '__main__':
	unittest.main()
