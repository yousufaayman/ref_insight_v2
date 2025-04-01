import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000
            ):
    
    logging.info("Start training")
    
    # Ensure the model save directory exists
    os.makedirs(best_model_path, exist_ok=True)
    
    # Track best performance
    best_validation_score = 0
    best_model_path_full = None

    for epoch in range(epoch_start, max_epochs):
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )

        training_results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(training_results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid"
        )

        validation_results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(validation_results)

        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
            )

        test_results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(test_results)
        
        # Step the learning rate scheduler
        scheduler.step()

        # Save model periodically and track best model
        # Use leaderboard_value as the performance metric
        current_validation_score = validation_results['leaderboard_value']
        
        # Save the current model
        current_model_path = os.path.join(best_model_path, f'model_epoch_{epoch+1}.pth.tar')
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_results': training_results,
            'validation_results': validation_results,
            'test_results': test_results
        }
        torch.save(state, current_model_path)
        
        # Update best model if current validation score is better
        if current_validation_score > best_validation_score:
            best_validation_score = current_validation_score
            best_model_path_full = os.path.join(best_model_path, 'best_model.pth.tar')
            torch.save(state, best_model_path_full)
            logging.info(f"New best model saved with validation score: {best_validation_score}")
        
        # Log epoch results
        logging.info(f"Epoch {epoch+1} Results:")
        logging.info(f"Training Leaderboard Score: {training_results['leaderboard_value']}")
        logging.info(f"Validation Leaderboard Score: {validation_results['leaderboard_value']}")
        logging.info(f"Test Leaderboard Score: {test_results['leaderboard_value']}")
        
        # Optional: Early stopping or other termination conditions
        if epoch - epoch_start > 10 and best_validation_score < 20:
            logging.info("Performance not improving. Stopping training.")
            break
    
    pbar.close()
    
    # Final model save
    final_model_path = os.path.join(best_model_path, 'final_model.pth.tar')
    final_state = {
        'epoch': max_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_validation_score': best_validation_score
    }
    torch.save(final_state, final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # If a best model was found, log its path
    if best_model_path_full:
        logging.info(f"Best model saved to {best_model_path_full}")
    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for targets_offence_severity, targets_action, mvclips, action in dataloader:

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            
            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
          
        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss




# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):
    

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
           
    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            #mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values                    


        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file
