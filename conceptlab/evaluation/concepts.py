
from sklearn.metrics import f1_score,precision_recall_curve, auc,roc_curve, roc_auc_score

def concept_accuarcy (concepts,pred_concept,debug=True):
	n_concept=concepts.shape[1]
	loss_dict={}
	all_acc=0
	all_f1=0
	all_auprc=0
	all_auroc=0
	for c in range(n_concept):
		c_true = concepts[:, c]
		c_pred = pred_concept[:, c]

		# Calculate F1 score
		threshold = 0.5
		c_pred_bin = (c_pred >= threshold).astype(int)
		f1 = f1_score(c_true, c_pred_bin)
		all_f1+=f1

		# Calculate AUPRC
		precision, recall, _ = precision_recall_curve(c_true, c_pred)
		auprc = auc(recall, precision)
		all_auprc+=auprc

		# Calculate AUROC directly
		auroc = roc_auc_score(c_true, c_pred)
		all_auroc+=auroc

		correct_predictions = (c_pred_bin == c_true).sum().item()
		total_predictions = c_true.shape[0]
		accuracy = correct_predictions / total_predictions
		all_acc+=accuracy

		if debug:
			loss_dict["test_concept_"+str(c) + "_f1"] = f1
			loss_dict["test_concept_"+str(c) + "_auprc"] = auprc
			loss_dict["test_concept_"+str(c) + "_auroc"] = auroc
			loss_dict["test_concept_"+str(c) + "_acc"] = accuracy

	loss_dict["test_avg_concept_auprc"] = all_auprc/n_concept
	loss_dict["test_avg_concept_auroc"] = all_auroc/n_concept
	loss_dict["test_avg_concept_f1"] = all_f1/n_concept
	loss_dict["test_avg_concept_acc"] = all_acc/n_concept

	return loss_dict