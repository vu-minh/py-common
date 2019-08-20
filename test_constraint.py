'''Test auto-generated constraints util functions
'''

import unittest
import math
import numpy as np

from dataset import dataset
from dataset import constraint


def nCr(n, r):
    """Calculate `n` choose `r`"""
    f = math.factorial
    n_combinations = f(n) / f(r) / f(n-r)
    return int(n_combinations)


class AutoGeneratedConstraintsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset.set_data_home("./data")
        test_dataset_name = "DIGITS"
        X_original, X, y = dataset.load_dataset(test_dataset_name)

        cls.dataset_name = test_dataset_name
        cls.data = X
        cls.labels = y

    def test_data_home(self):
        self.assertEqual(dataset.get_data_home(), './data')

    def test_number_of_autogenerated_constraints(self):
        n_expected = 50
        sim_links = constraint.gen_similar_links(self.labels, n_links=50)
        self.assertEqual(n_expected, len(sim_links), "number of sim-link error")
        dis_links = constraint.gen_dissimilar_links(self.labels, n_links=50)
        self.assertEqual(n_expected, len(dis_links), "number of dis-link error")

    def test_similar_links(self):
        sim_links = constraint.gen_similar_links(self.labels, n_links=50,
                                                 include_link_type=False)
        for p1, p2 in sim_links:
            label1, label2 = self.labels[[p1, p2]]
            self.assertEqual(label1, label2, "sim-link same label error")

    def test_dissimilar_links(self):
        dis_links = constraint.gen_dissimilar_links(self.labels, n_links=50,
                                                    include_link_type=False)
        for p1, p2 in dis_links:
            label1, label2 = self.labels[[p1, p2]]
            self.assertNotEqual(label1, label2, "dis-link different label error")


class ConstraintsFromPartialLabelsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset.set_data_home("./data")
        test_dataset_name = "DIGITS"
        X_original, X, y = dataset.load_dataset(test_dataset_name)

        cls.dataset_name = test_dataset_name
        cls.data = X
        cls.labels = y

        cls.n_labels_each_class = 5
        partial_labels = constraint.pick_random_labels(
            labels=y, n_labels_each_class=cls.n_labels_each_class, seed=42)
        cls.partial_labels = partial_labels

    def test_data_home(self):
        self.assertEqual(dataset.get_data_home(), "./data")

    def test_number_of_selected_labels(self):
        for class_id in np.unique(self.labels):
            self.assertEqual(self.n_labels_each_class,
                             len(self.partial_labels[class_id]))

    def test_selected_indices_same_labels(self):
        for class_id, selected_indices in self.partial_labels.items():
            selected_labels = self.labels[selected_indices]
            self.assertTrue(np.all(selected_labels == selected_labels[0]))

    def test_selected_labels_have_correct_class_id(self):
        for class_id, selected_indices in self.partial_labels.items():
            self.assertTrue(np.all(self.labels[selected_indices] == class_id))

    def test_number_of_generated_sim_links_from_partial_labels(self):
        n_labels_each_class = 5
        C_2_5 = nCr(n=5, r=2)
        n_classes = 10
        n_expected_sim_links = n_classes * C_2_5

        partial_labels = constraint.pick_random_labels(self.labels, n_labels_each_class)
        sim_links = constraint.generate_all_sim_links_from_partial_labels(partial_labels)
        self.assertEqual(n_expected_sim_links, len(sim_links))

    def test_generated_sim_links_from_partial_labels(self):
        sim_links = constraint.generate_all_sim_links_from_partial_labels(self.partial_labels)
        for p1, p2 in sim_links:
            label1, label2 = self.labels[[p1, p2]]
            self.assertEqual(label1, label2, "sim-link same label error")

    def test_number_of_generated_dis_links_from_partial_labels(self):
        n_labels_each_class = 5
        n_classes = 10
        n_pairs_of_two_different_classes = nCr(n=n_classes, r=2)
        n_links_per_each_class_pair = n_labels_each_class ** 2
        n_expected_dis_links = (n_pairs_of_two_different_classes *
                                n_links_per_each_class_pair)

        partial_labels = constraint.pick_random_labels(self.labels, n_labels_each_class)
        dis_links = constraint.generate_all_dis_links_from_partial_labels(partial_labels)
        self.assertEqual(n_expected_dis_links, len(dis_links))

    def test_generated_dis_links_from_partial_labels(self):
        dis_links = constraint.generate_all_dis_links_from_partial_labels(self.partial_labels)
        for p1, p2 in dis_links:
            label1, label2 = self.labels[[p1, p2]]
            self.assertNotEqual(label1, label2, "dis-link different label error")

    def test_reproduce_with_random_seed(self):
        seed = 4224
        partial_labels1 = constraint.pick_random_labels(self.labels, 5, seed)
        partial_labels2 = constraint.pick_random_labels(self.labels, 5, seed)
        self.assertEqual(partial_labels1, partial_labels2)

    def test_different_result_with_different_seed(self):
        partial_labels1 = constraint.pick_random_labels(self.labels, 5, seed=42)
        partial_labels2 = constraint.pick_random_labels(self.labels, 5, seed=24)
        self.assertNotEqual(partial_labels1, partial_labels2)


if __name__ == "__main__":
    unittest.main()