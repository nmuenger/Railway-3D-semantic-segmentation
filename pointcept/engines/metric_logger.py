import torch
import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu, intersection_and_union
import torch.distributed as dist
import numpy as np 

def update_distancewise_metric(coords, pred, segment, cfg, storage):
    # From here implemented distance-wise metrics
    dist_to_sensor = torch.norm(coords[:,0:2],dim=1) # Don't take z into account for distance to sensor
    
    min_dist = 0 # Originally set in distance to 0
    for max_dist in [20,40,60,80,100,140,180,220,1000,-1]: # Compute intersection, union for different distance ranges. -1 indicates all points
        if max_dist == -1: # If evaluate on all point, simply select all the points
            mask_dist = torch.ones_like(dist_to_sensor, dtype=torch.bool) # Mask is entirely True for that case
        else:
            mask_dist = (dist_to_sensor >= min_dist) & (dist_to_sensor < max_dist) # lower inclusive, upper exclusive

        if type(pred) is np.ndarray: # For some reason they use cpu numpy in tester and gpu in evaluator

            mask_dist = mask_dist.cpu().numpy()

            intersection, union, target = intersection_and_union(
                pred[mask_dist],
                segment[mask_dist],
                cfg.data.num_classes,
                cfg.data.ignore_index,
            )

        else:
            intersection, union, target = intersection_and_union_gpu(
                pred[mask_dist],
                segment[mask_dist],
                cfg.data.num_classes,
                cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (intersection.cpu().numpy(),union.cpu().numpy(),target.cpu().numpy(),)
        
        # Here there is no need to sync since sync happened in dist.all_reduce
        storage.put_scalar("val_intersection"+str(max_dist), intersection)
        storage.put_scalar("val_union"+str(max_dist), union)
        storage.put_scalar("val_target"+str(max_dist), target)

        min_dist = max_dist # Update min dist for next step

def log_metric(cfg, storage, logger, writer, current_epoch):
    # Compute the metric values for the distance ranges based on value found in previous loop
    for logger_dist_range, max_dist in zip(["0-20","20-40","40-60","60-80","80-100","100-140","140-180","180-220","220-1000","total"],[20,40,60,80,100,140,180,220,1000,-1]):

        
        intersection = storage.history("val_intersection"+str(max_dist)).total
        union = storage.history("val_union"+str(max_dist)).total
        target = storage.history("val_target"+str(max_dist)).total
        iou_class = intersection / (union + 1e-10)
        recall_class = intersection / (target + 1e-10) # This was originally named accuracy but i changfed it for Recall as in my opinion it fit better the definition. The variable "all_acc" is the "proper" accuracy
        precision_class = intersection/(union-target+intersection+1e-10)
        # Evaluate solely on the class that we specify
        m_iou_subset_evaluated = np.mean(iou_class[cfg.data.evaluated_class]) 
        m_recall_subset_evaluated = np.mean(recall_class[cfg.data.evaluated_class])
        m_precision_subset_evaluated = np.mean(precision_class[cfg.data.evaluated_class])
        all_acc_subset_evaluated = sum(intersection[cfg.data.evaluated_class]) / (sum(target[cfg.data.evaluated_class]) + 1e-10)
        # Evaluate on ALL classes
        m_iou = np.mean(iou_class) 
        m_recall = np.mean(recall_class)
        m_precision = np.mean(precision_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        logger.info(
            "Val result for distance {:.1f}: mIoU/mRecall/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                max_dist, m_iou, m_recall, all_acc
            )
        )
        for i in range(cfg.data.num_classes):
            logger.info(
                "Class_{idx}-{name}, distance {max_dist:.1f} Result: iou/recall/precision {iou:.4f}/{recall:.4f}/{precision:.4f}".format(
                    idx=i,
                    name=cfg.data.names[i],
                    max_dist=max_dist,
                    iou=iou_class[i],
                    recall=recall_class[i],
                    precision=precision_class[i]
                )
            )
            writer.log({"epoch":current_epoch,
                                    "per_class/iou/"+ logger_dist_range + "/" + cfg.data.names[i]:iou_class[i],
                                    "per_class/recall/"+ logger_dist_range + "/" + cfg.data.names[i]:recall_class[i],
                                    "per_class/precision/"+logger_dist_range + "/" + cfg.data.names[i]:precision_class[i]})
        
        if writer is not None:
            writer.log({"epoch":current_epoch,
                                    "val/"+ logger_dist_range +"/mIoU": m_iou, 
                                    "val/"+ logger_dist_range +"/mRecall": m_recall, 
                                    "val/"+ logger_dist_range +"/mPrecision": m_precision,
                                    "val/"+ logger_dist_range + "/allAcc": all_acc})
            writer.log({"epoch":current_epoch,
                                    "val/"+ logger_dist_range +"/mIoU_subset_evaluated": m_iou_subset_evaluated, 
                                    "val/"+ logger_dist_range +"/mRecall_subset_evaluated": m_recall_subset_evaluated, 
                                    "val/"+ logger_dist_range +"/mPrecision_subset_evaluated": m_precision_subset_evaluated,
                                    "val/"+ logger_dist_range + "/allAcc_subset_evaluated": all_acc_subset_evaluated})

    return m_iou