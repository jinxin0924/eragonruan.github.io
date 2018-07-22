---
layout:     post
title:      "Optimizer in Tensorflow"
date:       2018-07-20 14:48:00
author:     "JxKing"
header-img: "img/post-bg-001.jpg"
catalog: true
tags:
    - optimizer, tensorflow
---


## Optimizer in tensorflow

## 前传

写Optimizer系列文章，是因为去年2017年在华为做深度学习相关工作时，学习实现了许多基于tensorflow的optimizer的，开源了其中两个分布式的optimizer，并且合入了tf社区，还有个相关专利，做的工作还算出色。由于近期离开华为，加入了蚂蚁，为了纪念，也为了以后的学习，所以在不涉及保密工作的情况下，记录下学习心得。

系列文章将分为两篇，一篇讲原理，而本篇讲基于tensorflow的实现。

本篇文章从实现角度，将optimizer分为base optimizer、wrapper optimizer两部分展开。base optimizer与wrapper optimizer，顾名思义，wrapper optimizer是对base optimizer的一个包装，主要增加一些分布式相关的操作。



## 1. Base Optimizer

### 1.1 Optimizer基类

TF的optimizer都继承自Optimizer这个类，这个类的方法非常多，其中我个人认为重要的几个方法是 minimize、compute_gradients、apply_gradients、slot系列方法这几个方法。

1. **compute_gradients**: 传入loss，如果不传入var_list，那么默认就是所有trainable的variable，返回的是 list of (gradient, variable) pairs。
2. **apply_gradients**: 传入 (gradient, variable) pairs，将梯度apply到变量上。具体梯度如何更新到变量，由 \_apply_dense、\_resource_apply_dense、\_apply_sparse、\_resource_apply_spars这四个方法实现。
3. minimize：就是compute_gradients + apply_gradients
4. **slot系列**: 输入变量和name，得到的是一个 trainable=False的变量，用来记录optimizer中的中间值，比如在Momentum中，记录momentum。

### 1.2 Base Optimizer

base optimizer在继承Optimizer之后，只需要实现：

1. slot：如果有中间变量需要存储，则需要在初始时创建
2. \_apply_dense、\_resource_apply_dense、\_apply_sparse、\_resource_apply_spars这四个方法。这四个方法为了追求速度，一般都是用c++在`tensorflow/core/kernels/training_ops.cc` 中实现

以 MomentumOptimizer为例：

1. ```python
   # 创建momentum中间量
   def _create_slots(self, var_list):
   
   # 将类创建时可能传入的参数learning_rate、momentum转变为tensor
   def _prepare(self):
   
   # 调用c++实现的apply计算
   def _apply_dense(self, grad, var):
   ...
   ```

   

## 2. Wrapper Optimizer

就如之前所说，wrapper optimizer一般会传入base optimizer，用base optimizer来做具体的variable更新操作，而wrapper optimizer一般做些分布式相关工作。

在介绍wrapper optimizer之前，先大概介绍下深度学习中的分布式。深度学习中分布式一般采用**ps + 数据并行**，模型变量存在ps上，而模型的计算图与数据放在每个worker上，在更新时，每个worker会先从ps上拉取变量的值，进行前向与后向计算，完成后向计算后，会将梯度上传到ps上，由ps负责更新。而根据更新方式，可以分为同步与异步两种。同步时，ps会在收集到所有worker的梯度之后更新，而异步时，ps每收到一个梯度值，更新一次。

### 2.1 SyncReplicasOptimizer

SyncReplicasOptimizer是tensorflow官方实现的一个分布式opt，它在同步的基础上，加了一个*replicas_to_aggregate*参数，这个参数可以小于*total_num_replicas*，使得ps只要收集到*replicas_to_aggregate*个梯度就可以更新，避免了木桶效应导致的速度降低。具体的算法介绍可以参考[REVISITING DISTRIBUTED SYNCHRONOUS SGD](https://arxiv.org/pdf/1604.00981.pdf)。

我们可以看看代码，来深入了解下这个opt是如何实现的。

```python
  def __init__(self,
             opt, # 传入base_opt
             replicas_to_aggregate, # 收集到多少份梯度后更新
             total_num_replicas=None,
             variable_averages=None,
             variables_to_average=None,
             use_locking=False,
             name="sync_replicas"):
    ...
               

  def compute_gradients(self, *args, **kwargs): 
    # 用base_opt计算梯度
    return self._opt.compute_gradients(*args, **kwargs)
```

有上述可以看到，具体的计算梯度是由base_opt完成，其实具体的apply_gradients也是由base_opt完成的，SyncReplicasOptimizer的apply_gradients做的是同步操作。

```python
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ...
    local_anchor = control_flow_ops.no_op()
    with ops.colocate_with(local_anchor):
      self._local_step = variable_scope.variable(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=global_step.dtype.base_dtype,
          name="sync_rep_local_step")
```

上面这段代码是在本地创建local_step，local_step是为了记录此次更新的timestamp，而local_anchor的作用是强行将local_step这个变量放在worker上。记得replicas_to_aggregate是可能比total_num_replicas小的，那就意味着有几个速度慢的worker的梯度，是要被废弃的，因此local_step会与global_step作对比，如果local_step比global_step小，那么这份梯度就会被放弃。

接下来是主要的同步代码，代码相对而言长一点：

```python
    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        # 所有的var都会分布在ps上，ps可能是多个的，所以下面的操作绑定在每个var对应的device上，才能确保       速度
        with ops.device(var.device):
          # Dense gradients.
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          elif isinstance(grad, ops.Tensor):
            # 每个var创建一个accumulator
            grad_accum = data_flow_ops.ConditionalAccumulator(
                grad.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/grad_accum")
            # 梯度与local_step传入accumulator
            train_ops.append(grad_accum.apply_grad(
                grad, local_step=self._local_step))
                
            # 从accumulator中取出replicas_to_aggregate份梯度的均值，只要accumulator有，就会被取出
            aggregated_grad.append(grad_accum.take_grad(
                self._replicas_to_aggregate))
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_indexed_slices_grad(
                grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))
      
      # 取出的梯度均值与变量重新组成grads_and_vars
      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      with ops.device(global_step.device), ops.name_scope(""):
      # base_opt用梯度均值实际更新变量
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)
```

在上面这段代码中，每个worker都将自己的梯度放入accumulator中，然后accumulator取出replicas_to_aggregate份梯度的均值，用base_opt更新变量。

接下来还有最后一段代码，是用queue来做控制，不停的尝试从accumulator中取出平均梯度。

```python
     # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
      # 同步队列，放入global_step
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
        train_op = state_ops.assign(self._local_step, token)

        # 与update_op强依赖，利用队列强制base_opt做更新 
        with ops.control_dependencies([update_op]):
          tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))
        
        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
```

为什么要额外用队列呢？我们来分析下worker所做的事情，worker首先在计算完梯度之后，将梯度传递到accumulator（这些操作被定义为train_op），这时不能像往常那样由该worker执行 update_op（将平均梯度用base_opt更新），因为这是要等待其他worker完成梯度上传的，需要等待，而worker要去做计算操作，不然浪费算力，得不偿失。因此 需要另起一个线程QueueRunner，专门来执行update_op。

总体来说，这些op的关系如下：

* worker：在train_op完成之后，从 sync_token_queue中dequeue出一个token（global_step的任务），然后将这个token赋值给自己的local_step，相当于再领取一个任务。
* QueueRunner:  QueueRunner不断尝试执行 sync_op(往队列里填global_step)，而这个sync_op**依赖**于update_op，所以update_op会被先执行，然后global_step也被更新，新的global_step加入sync_token_queue，填充任务。

SyncReplicasOptimizer提供的用**accumulator和队列** 来做同步机制的方式，非常值得学习，之后的分布式optimizer都可以借鉴。



### 2.2 ModelAverage Opitmizer

[ModelAverage]( <https://arxiv.org/pdf/1410.7455v8.pdf>) 是一种特别的同步优化器。一般的同步模式，每次迭代，worker都需要从ps上拉取变量，然后计算出梯度之后上传到ps，这里面存在大量的网络通讯，而且由于是同步模式，极有可能造成阻塞。而model average的worker会先自己更新，在特定步数之后，对所有worker的变量求其平均值，然后更新。

在我的实现中，是在创建变量时，自动创建两份变量，一份local放在本地worker，另一份global放在ps上，训练时更新local，到达特点步数之后，将平均值更新到global上，然后global的值assign给local，完成更新。这是我提到TensorFlow社区的[pr](https://github.com/tensorflow/tensorflow/pull/15299), 具体实现可以参考这里。

下面我们来详细看下代码。

首先看下如何创建两份变量，这是通过CustomGetter实现的。custom_getter可以通过` with tf.variable_scope('',custom_getter=my_custom_getter):` 这种方式控制变量的创建。而我实现的modelaverage 的custom_getter如下：

```python
class ModelAverageCustomGetter(object):
  # 传入worker_device，是为了将local_var固定在worker上
  def __init__(self, worker_device):
    """Create a new `ModelAverageCustomGetter`.
    Args:
      worker_device: String.  Name of the `worker` job.
    """
    self._worker_device = worker_device
    # local_var到global_var的映射
    self._local_2_global = {}

  # 在创建变量时，会调用这个函数
  def __call__(self, getter, name, trainable, collections, *args, **kwargs):
    #只对trainable=True的变量进行处理
    if trainable:
      # 将local_var固定在worker上
      with ops.device(self._worker_device):
        # 创建local_var,将collection改成local_variables
        local_var = getter(
            name,
            trainable=True,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            *args,
            **kwargs)
      # 创建global_var，放在ps上
      global_variable = variable_scope.variable(
          name="%s/%s" % (GLOBAL_VARIABLE_NAME, name),
          initial_value=local_var.initialized_value(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES])

      self._local_2_global[local_var] = global_variable
      return local_var
    else:
      return getter(name, trainable, collections, *args, **kwargs)
```

通过ModelAverageCustomGetter，我们就可以将正常训练用的local_var放在worker上，另有一份global_var放在ps上。

接下来看看ModelAverage opt如何实现。

首先是这个opt的初始化：

```python
class ModelAverageOptimizer(optimizer.Optimizer):
  def __init__(self,
               opt, # 传入base_opt, worker自己更新用的opt
               num_worker,# woker数量，需要这个值来做同步
               is_chief, # chief worker需要额外做些工作
               ma_custom_getter, # custom_getter，为了获取local_2_global的映射
               interval_steps=100, # 特定步数
               use_locking=True,
               name="ModelAverageOptimizer"):
```

这个opt的compute_gradients就是调用 base_opt的compute_gradients，所以我们直接看apply_gradients函数。

```python
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # 利用base_opt更新local变量，同时每更新一次，local_step +=1
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      raise ValueError("Global step is required")

    apply_updates = self._opt.apply_gradients(grads_and_vars)
    with ops.control_dependencies([apply_updates]):
      local_update = state_ops.assign_add(
          self._local_step, 1, name="local_step_update").op

    # 到达特点步数之后，如何更新global_varibales，具体实现我们放在下面
    def _update_global_variables():
        ... 
    
    # 在完成local更新之后，做判断，是否达到了特定步数，如果是，则执行_update_global_variables，否则什么都不做，no_op,
    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(
          math_ops.mod(self._local_step, self._interval_steps), 0)
      conditional_update = control_flow_ops.cond(
          condition, _update_global_variables, control_flow_ops.no_op)
    return conditional_update
```

上述是opt的apply_gradients的整体思路，接下来关键就是如何来实现_update_global_variables这个函数。

```python
    def _update_global_variables(): 
      # 分别拿到local_vars和global_vars
      local_vars = [v for g, v in grads_and_vars if g is not None]
      global_vars = [self._local_2_global[v] for v in local_vars]
      # 创建同步队列sync queue
      with ops.colocate_with(global_step):
        sync_queue = data_flow_ops.FIFOQueue(
            -1, [dtypes.bool], shapes=[[]], shared_name="sync_queue")
      train_ops = []
      aggregated_vars = []
      with ops.name_scope(None, self._name + "/global"):
        for var, gvar in zip(local_vars, global_vars):
          with ops.device(gvar.device):
            if isinstance(var._ref(), ops.Tensor):
              # 参考SyncReplicaOptimizer，将每个worker的local_var放入accumulator  
              var_accum = data_flow_ops.ConditionalAccumulator(
                  var.dtype,
                  shape=var.get_shape(),
                  shared_name=gvar.name + "/var_accum")
              train_ops.append(
                  var_accum.apply_grad(var._ref(), local_step=global_step))
              # 取出num_worker份local_var的平均值
              aggregated_vars.append(var_accum.take_grad(self._num_worker))
            else:
              raise ValueError("Unknown local variable type!")
            self._accumulator_list.append((var_accum, gvar.device))
      # chief worker负责将local_var的平均值赋值给global_var，然后在同步队列中加入num_worker-1份数据
      if self._is_chief:
        update_ops = []
        with ops.control_dependencies(train_ops):
          for avg_var, gvar in zip(aggregated_vars, global_vars):
            with ops.device(gvar.device):
              update_ops.append(state_ops.assign(gvar, avg_var))
          with ops.device(global_step.device):
            update_ops.append(state_ops.assign_add(global_step, 1))
        with ops.control_dependencies(update_ops), ops.device(
            global_step.device):
          tokens = array_ops.fill([self._num_worker - 1],
                                  constant_op.constant(False))
          sync_op = sync_queue.enqueue_many(tokens)
     # 其它worker的sync_op是从同步队列中取出一份token，当chief_worker没有完成global_var更新时，同步队列为空，因而其余worker只能等待，从而完成同步
      else:
        with ops.control_dependencies(train_ops), ops.device(
            global_step.device):
          sync_op = sync_queue.dequeue()
      # 在完成global_var更新之后，所有worker将global_var赋值给local_var
      with ops.control_dependencies([sync_op]):
        local_update_op = self._local_vars_update(local_vars)
      return local_update_op
```

整个思路很简单：

1. 所有worker将各自的local_var塞到accumulator里面
2. chief worker负责将 num_worker份local_var的平均值从accumulator取出，然后赋值给global_var
3. Chief worker完成global_var更新之后，往同步队列填充token
4. 其余worker在完成1之后，执行从同步队列取token操作。如果chief worker没有完成global_var更新操作，那么此时同步队列为0，其余worker只能等待，直到chief worker完成global_var的更新
5. 所有worker执行global_var赋值给local_var的更新操作



这些就是ModelAverageOptimizer的主要逻辑代码了，其余还有些初始化的工作就不详细介绍了，更具体内容可以看`/tensorflow/contrib/opt/python/training/model_average_optimizer.py`。

在这样的实现下，在没有达到特点步数时，由于用于更新的local_var就在本地，所以woker的更新无需网络通信，迭代的速度非常快。



### 2.3 ElasticAverage Opitmizer

ElasticAverage Optimizer也是类似于ModelAverage的一种opt，它的具体算法可以参考：[Deep learning with Elastic Averaging SGD](https://arxiv.org/pdf/1412.6651.pdf)，也可以看我接下来聚焦在opt算法的文章。

这是我提的[pr](https://github.com/tensorflow/tensorflow/pull/13012)，这是我第一次向Tensorflow提交pr，改动非常多，学到了非常多东西。

这里的实现其实跟ModelAverage的大同小异，也是通过custom_getter创建local_vars和global_vars，然后通过同步队列控制同步操作。具体代码不细讲了，如果有疑问，可以直接在github上联系我，或者联系我邮箱：jinxin7120@gmail.com.



### 2.4 其它分布式Optimizer

总之实现分布式Optimizer有三宝：**同步队列、accumulator、custom_getter**。掌握了这三种之后，实现各种分布式optimizer 都会容易很多。举一些例子：

* [Softsync](https://arxiv.org/pdf/1511.05950.pdf)：可以在SyncReplicaOptimizer的基础上修改accumulator实现。
* [Distributed KFAC](https://jimmylba.github.io/papers/nsync.pdf)：可以将base_opt设置成kfac，然后在apply_gradients函数中，添加各种完成，完成paper提出的opt。
* [LARS](https://arxiv.org/abs/1708.03888)：可以通过accumulator与custom_getter将梯度累积在本地多次，简单实现超大batch_size的更新。
* [Deep Gradient Compression](https://arxiv.org/pdf/1712.01887.pdf): 也是类似





