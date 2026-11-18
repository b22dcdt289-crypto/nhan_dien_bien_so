"""Behavior checks for the supported 0/O and motorcycle decision cases."""
import unittest
import numpy as np
import train_model as t
from evaluate_refined import proposals_refined
from evaluate_robust import fair_plate_scores

class PolicyBehavior(unittest.TestCase):
    def sample(self,position,values,n=9):
        p=np.full((n,len(t.CLASSES)),1e-6,dtype=np.float64)
        for char,val in values.items():p[position,t.CLASSES.index(char)]=val
        return proposals_refined(p,[{'start':0,'count':n}])
    def test_motorcycle_fourth_character_may_be_letter_or_digit(self):
        for char in ('A','7'):
            _,chosen,_,_,_=self.sample(3,{char:.9})
            self.assertEqual(t.CLASSES[chosen[3]],char)
    def test_ambiguous_zero_o_needs_both_visual_support_and_position(self):
        raw,chosen,score,margin,ratio=self.sample(2,{'0':.51,'O':.47})
        self.assertEqual(t.CLASSES[raw[2]],'0');self.assertEqual(t.CLASSES[chosen[2]],'O')
        self.assertGreater(score[2],.9);self.assertGreater(ratio[2],.7)
        _,chosen,_,_,_=self.sample(5,{'0':.47,'O':.51})
        self.assertEqual(t.CLASSES[chosen[5]],'0')
    def test_strong_conflict_is_not_silently_promoted(self):
        raw,chosen,score,margin,ratio=self.sample(2,{'0':.97,'O':.02,'A':.01})
        self.assertNotEqual(raw[2],chosen[2]);self.assertLess(ratio[2],.7)
    def test_unrestricted_zero_o_tie_is_uncertain(self):
        _,_,_,margin,_=self.sample(3,{'0':.51,'O':.48})
        self.assertLess(margin[3],.05)
    def test_wrong_length_accepted_is_counted_as_wrong_not_hidden(self):
        rows=[{'start':0,'count':8,'truth_posthoc':'12A312345','segmentation':{'ok':True}}]
        correct,accepted,wrong=fair_plate_scores(rows,np.zeros(8,dtype=int),np.ones(8,dtype=bool))
        self.assertEqual((correct,accepted,wrong),(0,1,1))

if __name__=='__main__':unittest.main()
