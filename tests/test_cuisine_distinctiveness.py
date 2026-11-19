import math
import unittest

from src.cuisine_distinctiveness import score_recipes


class DistinctivenessTests(unittest.TestCase):
    def test_scored_recipe_does_not_make_its_own_pair_distinctive(self):
        # The first A recipe's (x,y) pair is otherwise absent from A but
        # appears in one of two B recipes. Without leave-one-out it would
        # receive a spuriously positive A score.
        scores, signatures = score_recipes(
            [1, 2, 3, 4],
            ["A", "A", "B", "B"],
            [[0, 1], [0, 2], [0, 1], [1, 2]],
            ["x", "y", "z"],
            min_pair_support=1,
        )
        self.assertEqual(scores.loc[0, "pair_distinctiveness"], -0.5)
        self.assertEqual(scores.loc[1, "pair_distinctiveness"], 0.0)
        self.assertEqual(scores.loc[0, "ingredient_distinctiveness"], -0.25)
        self.assertEqual(signatures.loc[signatures.cuisine.eq("A") & signatures.ingredient_b.eq("y"), "recipes_in_cuisine"].iloc[0], 1)

    def test_recipe_with_one_retained_ingredient_has_no_pair_score(self):
        scores, _ = score_recipes(
            [1, 2, 3, 4], ["A", "A", "B", "B"],
            [[0], [0, 1], [0, 1], [1, 2]], ["x", "y", "z"],
        )
        self.assertEqual(scores.loc[0, "retained_pairs"], 0)
        self.assertTrue(math.isnan(scores.loc[0, "pair_distinctiveness"]))


if __name__ == "__main__":
    unittest.main()
