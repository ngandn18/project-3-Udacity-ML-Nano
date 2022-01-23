# Image Classification using AWS SageMaker

This project guide me to use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

This is done on the provided ***dog breed classication data set***. This dataset has 133 classes.

## Project Set Up and Installation

Complete two files ***hpo.py*** and ***train_model.py*** and test to make sure they are ok. We can use our private machine to prepare them.

Enter AWS through the gateway in the course and open SageMaker Studio. 

Download the starter files.

Open the notebook ***train_and_deploy.ipynb*** to run step by step.

Download and Make the dataset available.

Training and save the model to reuse in refenrence process. The file ***train_model.py*** is very important to train model efficiently. When the training is complete, the model is saved in the compressed file ***model.tgz***.

## Dataset

I get the dataset from the notebook ***train_and_deploy.ipynb***

***`!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip`***

***`!unzip -q dogImages.zip`***

### Access

I upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.

`inputs = sagemaker_session.upload_data(path=datapath, bucket=data_bucket, key_prefix=datapath)`

## Hyperparameter Tuning

I choose resnet34 because of its simplicity and its efficiency.

The list of hyperparameters and their ranges used for the hyperparameter search


`hyperparameter_ranges = {`

`    "lr": ContinuousParameter(0.001, 0.003),`
    
`    "batch_size": CategoricalParameter([32, 64, 128])`

`}`



File ***hpo.py*** is attached to show my code for executing Hyperparameter Tunning Job.

This is [Hyperparameter Tunning Job Image](images/tunning_job.png)

This is [Best model training hyperparameters image](images/best_hyperparameters.png)

I want to be sure that the values of best hyperparameter are used.

I print it out to examine and preprocess before I can use them in training job with estimator.

`best_hypers = best_estimator.hyperparameters()`

`print(best_hypers)`

`batch_size = best_hypers['batch_size'].replace('"','') `

`lr = best_hypers['lr']`

`print(type(batch_size), batch_size, type(lr), lr)`

`hyperparameters = {'batch_size': batch_size,`

`                   'lr': lr} # Training with best parameters`

[Image Get and retrieve the best hyperparameters](images/get_best_hyperparameters.png)


## Debugging and Profiling

Setting up debugging and profiling rules and hooks successfully.

`rules = [`

`    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),`

`    Rule.sagemaker(rule_configs.vanishing_gradient()),`

`    Rule.sagemaker(rule_configs.overfit()),`

`    Rule.sagemaker(rule_configs.overtraining()),`

`    Rule.sagemaker(rule_configs.poor_weight_initialization()),`

`    Rule.sagemaker(base_config=rule_configs.class_imbalance(),`

`                   rule_parameters={"labels_regex": "CrossEntropyLoss_input_1"})`

`]`



`profiler_config = ProfilerConfig(`

`    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)`

`)`


`debugger_hook_config = DebuggerHookConfig(`

`    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}`

`)`


Please see more in file ***train_and_deploy.html***

### Results

[Plot a debugging output](images/debugging_plot.png)


The profiler html/pdf file in the folder ***ProfilerReport*** is attached. 

## Model Deployment

**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The file ***train_model.py*** is used with the notebook ***train_and_deploy.ipynb***. 

I use model resnet34. I have tested with 10, 15, 10 and 30 epochs. With this data, I see that about 18 epochs is enough. When I test resnet50, it takes more time and can get the same accuracy as resnet34 with 15 epochs, then I decide that resnet34 is enough because there are only 133 classes in dataset.

I had a long time to solve the problem of creating endpoint for this model, after 13 days I have worked out.

The problem is from one line code: ***torch.save(model, os.path.join(args.model_dir, "model.pt"))***

I use script ***inference.py*** (attached in this submission) to creat model resnet34 (exact model in ***train_modal.py***) to load the saved model from s3 storage of the training job, and creat I three functions to support the prediction: ***predict_fn***, ***input_fn*** and ***output_fn***. 

The code cab be seen in the file ***train_and_deploy.html***, or can review the following image.

[Image of code create endpoint](images/endpoint_code.png)

[Image of my endpoint](images/endpoint.png)

Time to create endpoint is less than 219s.

After the first correct prediction with only 0.4s, I write a small code to get sequential 30 predictions from 30 random files of the data directory test.

The 1st time, the accuracy is about 57% in 4.96s. [It's here](images/prediction_1.png)

The 2st time, the accuracy is about 77% in 4.91s. [It's here](images/prediction_2.png)

The interesting predictions executes in a very short time.

## Standout Suggestions

In the previous project, there was a suggestion to calculate the successful predictions of each class. Now I try to do this idea in this project after I saw the time to predict is very short, then I write some code to do in such a way:

1. I prepare data by my local machine the csv files contain:

- the numbers of test files and train files and the numbers of each class in test directory.
- a file contain classid and name of all files in the directory test (836 files)

2. I write code to read data from the csv files.

3. I create new endpoint () to test all the files in directory test only in 109,9 s, less than 2 minutes and the accuracy is 59% (836 predictions). This model was trained with 10 epochs (I was afraid of out of aws credit :-( ), so the accuracy is low. [Here is the image](images/test_all.png). I tested on Kaggle with 20 epochs the accuracy was about 79%.

4. I write code to prepare the compatible plots. It shows that if the proportion of test files in class is larger, the successful proportion of prediction of the class is larger. We can see in the result file ****succ_class.csv****, the class contains more than 7 then the successful prediction of this class above 75% (attached in the submission zipfile) [Here is the plot](images/successful_predict_each_class.png)

5. I test the sns.relplot effectively by my pc, but it can not work with SG Studio Jupyter, so I save data to csv file and plot the final chart sns by my local machine. [Here sns.relplot.png](images/sns.relplot.png)

### Thank all Udacity members for all your helpful and valuable supports.
