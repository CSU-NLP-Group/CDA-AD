//local transformer_model = "/root/autodl-tmp/pretrained_models/bert-base-uncased/";
//local base_dir = "/root/autodl-tmp/ADReSS/data/";

//local transformer_model = "/home/data/pretrained_models/bert-base-uncased/";
//local base_dir = "/home/jwduan/workspace/ADReSS-Challenge/data/";

local transformer_model = "/home/duanjw/jwduan/pretrained_models/bert-base-uncased/";
local base_dir = "/home/duanjw/jwduan/ADReSS-Challenge/data/";


local transformer_dim = 768;

#是否是debug 模式
local debug = false;
#交叉验证折数，可以为5或者10
local num_folds = 10;
local num_epochs = 5;

local seed = 42;# (if debug then 42 else std.parseInt(std.extVar('seed')));

#梯度累积步数
local ga_steps = (if debug then 2 else std.parseInt(std.extVar('ga_steps')));
#学习率
local lr = (if debug then 1e-5 else std.parseJson(std.extVar('lr')));
#模型整体的dropout
local dropout = (if debug then 0.3 else std.parseJson(std.extVar('dropout')));
#margin loss中的margin
local margin = (if debug then 0.1 else std.parseJson(std.extVar('margin')));
#margin loss在整个损失中的权重
local alpha = (if debug then 0.5 else std.parseJson(std.extVar('alpha')));
#增加的案例数目
local num_aug = (if debug then 1 else std.parseInt(std.extVar('num_aug')));
#增加的负例数
local num_negs = (if debug then 5 else std.parseInt(std.extVar('num_negs')));
#删除的比例
local del_rate = (if debug then 0.3 else std.parseJson(std.extVar('del_rate')));
# 是使用avg_pooling还是bert_pooler，如果是avg_pooling
# 还会将确认是否使用非线性激活函数
local use_avg_pooling = (if debug then true else std.parseJson(std.extVar('use_avg')));
local use_feed_forward = (if debug then true else std.parseJson(std.extVar('use_fwd')));
#是否使用regularized dropout
local use_rdrop = (if debug then true else std.parseJson(std.extVar('use_rdrop')));
#在regularized dropout中会kl损失占的权重
local kl_weight = (if debug then 1.0 else std.parseJson(std.extVar('kl_weight')));


local bert_pooler =  {
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
};

local avg_pooling = {
    "type": "boe",
    "embedding_dim": transformer_dim,
    "averaged": true
};

local dataset_reader = {
    "type": "adreader",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    },

    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": 256,
      "tokenizer_kwargs":{
          "do_lower_case": true,
      }
    }
  };


{
  "type": "cross_validation",
//  "random_state": seed,
  "dataset_reader": dataset_reader{
    "num_aug": num_aug,
    "num_negatives":num_negs,
    "delete_rate": del_rate,
  },
  "validation_dataset_reader": dataset_reader{
    "is_training": false
  },

  "train_data_paths": [base_dir + "%dfold/fold-%d/train.txt" % [num_folds, i] for i in std.range(0, num_folds-1)],
  "val_data_paths": [base_dir + "%dfold/fold-%d/val.txt" % [num_folds, i] for i in std.range(0, num_folds-1)],
//  "test_data_path": base_dir + "ADRess_test_data.csv",
  "evaluate_on_test": true,
  "model": {
    "type": "ad_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "seq2vec_encoder": if use_avg_pooling then avg_pooling else bert_pooler,
    [if use_avg_pooling && use_feed_forward then "feedforward"]:{
        "input_dim": transformer_dim,
        "num_layers": 1,
        "hidden_dims": transformer_dim,
        "activations": {"type": "tanh"},
    },

    "dropout": dropout,
    "margin": margin,
    "alpha": alpha,
    "kl_weight": kl_weight,
    "use_rdrop": use_rdrop,
  },
  "data_loader": {
//    "type":"instance",
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 4
    }
  },

  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 16
    }
  },

  "trainer": {
    "num_epochs": num_epochs,
//    "patience": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
//      "warmup_steps": 20,
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "eps": 1e-8,
      "weight_decay": 0.1,
    },
//    [if !debug then "callbacks"]:[
//        {
//            "type":"wandb",
//            "should_log_learning_rate":true,
//            "project":"ADReSS",
//            "group":"10-fold-sweep",
//        }
//    ],

    "num_gradient_accumulation_steps": ga_steps,
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
}
