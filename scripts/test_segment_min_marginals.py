import itertools,math,unittest
from segment_min_marginals import min_sum_passage_marginals

class MarginalsTests(unittest.TestCase):
    def test_adjacent_one_row_ambiguity(self):
        scores=(1.00,1.01,3.00)
        def cost(j,done,count):
            return scores[j] if count else .70
        z=min_sum_passage_marginals(passage_count=3,row_count=1,transition=cost)
        self.assertEqual(z.rows[0].best_passage,0)
        self.assertEqual(z.rows[0].alternative_passage,1)
        self.assertAlmostEqual(z.rows[0].confidence,.01,places=6)
        self.assertEqual(z.adjacent_ambiguous_rows,(1,))
    def test_nonadjacent_structural_ambiguity(self):
        scores=(1.00,3.00,1.01)
        z=min_sum_passage_marginals(passage_count=3,row_count=1,
               transition=lambda j,done,count:scores[j] if count else .70)
        self.assertEqual(z.rows[0].alternative_passage,2)
        self.assertEqual(z.nonadjacent_ambiguous_rows,(1,))
    def test_matches_bruteforce_min_marginals(self):
        m,n,cap=3,3,2
        def cost(j,done,count):
            return (.4 if count==0 else
              .25*abs(done-j)+.13*count+.07*(j+1)*count)
        v=min_sum_passage_marginals(passage_count=m,row_count=n,
                                    max_lines_per_passage=cap,transition=cost)
        solutions=[]
        for counts in itertools.product(range(cap+1),repeat=m):
            if sum(counts)!=n:continue
            done=0; placements=[]
            total=0.
            for j,c in enumerate(counts):
                total+=cost(j,done,c)
                placements.extend([j]*c)
                done+=c
            solutions.append((total,placements))
        self.assertAlmostEqual(v.global_cost,min(x[0] for x in solutions))
        for i,item in enumerate(v.rows):
            minima={}
            for score,place in solutions:
                minima[place[i]]=min(score,minima.get(place[i],math.inf))
            ranked=sorted((score,j) for j,score in minima.items())
            self.assertEqual(item.best_passage,ranked[0][1])
            self.assertAlmostEqual(item.best_cost,ranked[0][0])
            self.assertEqual(item.alternative_passage,ranked[1][1])
            self.assertAlmostEqual(item.confidence,ranked[1][0]-ranked[0][0])
    def test_invalid_and_infeasible(self):
        with self.assertRaises(ValueError):
            min_sum_passage_marginals(passage_count=1,row_count=3,
                max_lines_per_passage=2,transition=lambda *a:1.)
