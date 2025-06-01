import math
from collections import Counter
import numpy as np

def bleu_stats(hypothesis, reference):
	# 计算BLEU的统计数据
	"""
	stats是列表，其中一共有10个元素，前两个分别是机器翻译译文和参考译文的长度，
	剩下的8个元素，每两个一组，是n-gram的相关信息，其中每一组，第一个数是统计的匹配数，第二个是候选译文的长度
	"""
	stats = []
	stats.append(len(hypothesis))
	stats.append(len(reference))

	for n in range(1, 5):
		s_ngrams = Counter(
			[tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
		)
		
		r_ngrams = Counter(
			[tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
		)

		stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
		stats.append(max([len(hypothesis) + 1 - n, 0]))
	return stats

def bleu(stats):
	# 根据n-gram统计数据计算BLEU
	if len(list(filter(lambda x: x == 0, stats))) > 0:
		return 0
	(c, r) = stats[0:2]
	log_bleu_prec = sum(
		[math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
	) / 4.
	return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypothesis, reference):
	stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
	for hyp, ref in zip(hypothesis, reference):
		stats += np.array(bleu_stats(hyp, ref))
	return 100 * bleu(stats)
