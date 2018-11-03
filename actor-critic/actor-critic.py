import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import os


def print_each_every(i,every, message):
    if i % every == 0:
        print(message)
        return True
    return False

def save_data_image(path,name,data):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(data,'bx')
    print(path[:-10]+'/'+name)
    fig.savefig(path[:-10]+'/'+name)   # save the figure to file
    plt.close(fig)        

class agent_episodic_continuous_action():
    def __init__(self,gamma,s_size,a_size,action_type,amount_of_data_to_memorize=100,optimizer="GRAD"):
        
        self.first_delta = True
        self.is_actor_brain_present = False
        self.is_critic_brain_present = False
        self.is_critic_decay_set = False
        self.is_actor_decay_set = False 
        self.is_action_discrete = False
        self.is_action_continous = False
        
        self.s_size = s_size
        self.a_size= a_size
        
        if action_type != "continous" and action_type != "discrete":
            raise Exception("only continous and dicrete values are supported")
        else:
            self.action_type = action_type # "continous" or "discrete"
            
            if self.action_type == "continous":
                self.is_action_continous = True
            else:
                self.is_action_discrete = True
                
        self.gamma = gamma
        self.I = 1
        
        self.weights_actor ={}
        self.biases_actor ={}
        
        self.weights_critic ={}
        self.biases_critic ={} 
        
        self.weights_critic_copy_1 ={}
        self.biases_critic_copy_1 ={} 

        self.weights_critic_copy_2 ={}
        self.biases_critic_copy_2 ={}         
        
        self.amount_of_data_to_memorize = amount_of_data_to_memorize
        self.internal_memory_counter = 0
        self.internal_step_counter = 0
        
        self.size_of_data = 2*s_size+5
        
        self.memories_holder = np.zeros([self.amount_of_data_to_memorize,self.size_of_data],dtype=np.float32) #limited discrete the a_size is 1
        
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32,name="input")
        self.state_in_future = tf.placeholder(shape=[None,s_size],dtype=tf.float32,name="input_future")
        self.reward_placeholder = tf.placeholder(shape=[None],dtype=tf.float32,name="reward")
        self.I_placeholder = tf.placeholder(shape=[None],dtype=tf.float32,name="ii")
        
        self.d_placeholder = tf.placeholder(shape=[None],dtype=tf.bool,name="dd")
        self.gamma_placeholder = tf.placeholder(shape=[None],dtype=tf.float32,name="gamma")
        
        self.alfa_regalutor = tf.placeholder(shape=[None],dtype=tf.float32,name="alfaregulator")
        
        
        if self.is_action_continous:
            self.action_placeholder = tf.placeholder(shape=[None,a_size],dtype=tf.float32,name="action")
            self.size_of_data = 2*s_size+4 + self.a_size 
            self.cont_offset = self.a_size
        elif self.is_action_discrete:   #discrete and not infinite 
            self.action_placeholder = tf.placeholder(shape=[None],dtype=tf.int32,name="action")
            self.size_of_data = 2*s_size+4 + 1
            self.cont_offset = 0
            
        self.memories_holder = np.zeros([self.amount_of_data_to_memorize,self.size_of_data])    
        self.set_data_indexes()

    def save_model(self,path,sess):
        self.saver.save(sess, path)

    def load_model(self,path,sess):
        self.saver.restore(sess, path)
    
    def weights_init_actor(self,hidd_layer,mean,stddev):   
        num_input = self.s_size
        
        if self.is_action_continous:        
            num_output = self.a_size*2
        elif self.is_action_discrete:
            num_output = self.a_size  
        else:
            raise Exception("only continous and dicrete values are supported")            
                         
        n_hidden_1 = hidd_layer[0]
        num_hidden_layers = len(hidd_layer)        
        self.weights_actor['h_{0}'.format(0)] = tf.Variable(tf.random_normal([num_input, n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='actor')
        self.biases_actor['b_{0}'.format(0)] = tf.Variable(tf.random_normal([n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='actor') 
        for i in range(num_hidden_layers):
            if i < num_hidden_layers-1:
                num_input = n_hidden_1
                n_hidden_1 = hidd_layer[i+1]
                self.weights_actor['h_{0}'.format(i+1)] = tf.Variable(tf.random_normal([num_input, n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='actor')
                self.biases_actor['b_{0}'.format(i+1)] = tf.Variable(tf.random_normal([n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='actor')
            else:
                self.weights_actor['h_{0}'.format("out")] = tf.Variable(tf.random_normal([n_hidden_1, num_output],mean=mean,stddev=stddev,dtype=tf.float32),name='actor')  
                self.biases_actor['b_{0}'.format("out")] = tf.Variable(tf.random_normal([num_output],mean=mean,stddev=stddev,dtype=tf.float32),name='actor')    

    def weights_init_critic(self,hidd_layer,mean,stddev):   
        num_input = self.s_size
        num_output = 1
        n_hidden_1 = hidd_layer[0]
        num_hidden_layers = len(hidd_layer)        
        self.weights_critic['h_{0}'.format(0)] = tf.Variable(tf.random_normal([num_input, n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='critic')
        self.biases_critic['b_{0}'.format(0)] = tf.Variable(tf.random_normal([n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='critic') 
        for i in range(num_hidden_layers):
            if i < num_hidden_layers-1:
                num_input = n_hidden_1
                n_hidden_1 = hidd_layer[i+1]
                self.weights_critic['h_{0}'.format(i+1)] = tf.Variable(tf.random_normal([num_input, n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='critic')
                self.biases_critic['b_{0}'.format(i+1)] = tf.Variable(tf.random_normal([n_hidden_1],mean=mean,stddev=stddev,dtype=tf.float32),name='critic')
            else:
                self.weights_critic['h_{0}'.format("out")] = tf.Variable(tf.random_normal([n_hidden_1, num_output],mean=mean,stddev=stddev,dtype=tf.float32),name='critic')  
                self.biases_critic['b_{0}'.format("out")] = tf.Variable(tf.random_normal([num_output],mean=mean,stddev=stddev,dtype=tf.float32),name='critic')    

    def weights_init_critic_copy_1(self,hidd_layer,mean,stddev):   
        num_input = self.s_size
        num_output = 1
        n_hidden_1 = hidd_layer[0]
        num_hidden_layers = len(hidd_layer)        
        self.weights_critic_copy_1['h_{0}'.format(0)] = tf.Variable(tf.zeros([num_input, n_hidden_1],dtype=tf.float32),name='criticcopy',trainable=True)
        self.biases_critic_copy_1['b_{0}'.format(0)] = tf.Variable(tf.zeros([n_hidden_1],dtype=tf.float32),name='criticcopy',trainable=True) 
        for i in range(num_hidden_layers):
            if i < num_hidden_layers-1:
                num_input = n_hidden_1
                n_hidden_1 = hidd_layer[i+1]
                self.weights_critic_copy_1['h_{0}'.format(i+1)] = tf.Variable(tf.zeros([num_input, n_hidden_1],dtype=tf.float32),name='criticcopy',trainable=True)
                self.biases_critic_copy_1['b_{0}'.format(i+1)] = tf.Variable(tf.zeros([n_hidden_1],dtype=tf.float32),name='criticcopy',trainable=True)
            else:
                self.weights_critic_copy_1['h_{0}'.format("out")] = tf.Variable(tf.zeros([n_hidden_1, num_output],dtype=tf.float32),name='criticcopy',trainable=True)  
                self.biases_critic_copy_1['b_{0}'.format("out")] = tf.Variable(tf.zeros([num_output],dtype=tf.float32),name='criticcopy',trainable=True) 


    def weights_init_critic_copy_2(self,hidd_layer,mean,stddev):   
        num_input = self.s_size
        num_output = 1
        n_hidden_1 = hidd_layer[0]
        num_hidden_layers = len(hidd_layer)        
        self.weights_critic_copy_2['h_{0}'.format(0)] = tf.Variable(tf.zeros([num_input, n_hidden_1],dtype=tf.float32),name='criticcopyfuture',trainable=True)
        self.biases_critic_copy_2['b_{0}'.format(0)] = tf.Variable(tf.zeros([n_hidden_1],dtype=tf.float32),name='criticcopyfuture',trainable=True) 
        for i in range(num_hidden_layers):
            if i < num_hidden_layers-1:
                num_input = n_hidden_1
                n_hidden_1 = hidd_layer[i+1]
                self.weights_critic_copy_2['h_{0}'.format(i+1)] = tf.Variable(tf.zeros([num_input, n_hidden_1],dtype=tf.float32),name='criticcopyfuture',trainable=True)
                self.biases_critic_copy_2['b_{0}'.format(i+1)] = tf.Variable(tf.zeros([n_hidden_1],dtype=tf.float32),name='criticcopyfuture',trainable=True)
            else:
                self.weights_critic_copy_2['h_{0}'.format("out")] = tf.Variable(tf.zeros([n_hidden_1, num_output],dtype=tf.float32),name='criticcopyfuture',trainable=True)  
                self.biases_critic_copy_2['b_{0}'.format("out")] = tf.Variable(tf.zeros([num_output],dtype=tf.float32),name='criticcopyfuture',trainable=True) 

    def create_critic_brain_copy_1(self,hidd_layer,hidd_act_fn,output_act_fn,mean,stddev):
        if self.is_actor_brain_present:            
            self.weights_init_critic_copy_1(hidd_layer,mean,stddev)        
            num_hidden_layers = len(hidd_layer)          
            if hidd_act_fn == "relu":      
                layer_h = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights_critic_copy_1['h_0']), self.biases_critic_copy_1['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.relu(tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format(i+1)]), self.biases_critic_copy_1['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format("out")]), self.biases_critic_copy_1['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format("out")]), self.biases_critic_copy_1['b_{0}'.format("out")]))                      
                self.output_critic_copy_1 = tf.reshape(layer_out_critic,[-1])
                self.output_critic_copy_1  = tf.stop_gradient(self.output_critic_copy_1)                
                self.critic_tvar_copy_1_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_copy_1_present = True  
            elif hidd_act_fn == "sigmoid":      
                layer_h = tf.nn.sigmoid(tf.add(tf.matmul(self.state_in, self.weights_critic_copy_1['h_0']), self.biases_critic_copy_1['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.sigmoid(tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format(i+1)]), self.biases_critic_copy_1['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format("out")]), self.biases_critic_copy_1['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic_copy_1['h_{0}'.format("out")]), self.biases_critic_copy_1['b_{0}'.format("out")]))                      
                self.output_critic_copy_1 = tf.reshape(layer_out_critic,[-1])
                self.output_critic_copy_1  = tf.stop_gradient(self.output_critic_copy_1)                
                self.critic_tvar_copy_1_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_copy_1_present = True                 
        else:
            print("please create actor brain first")
            
    def create_critic_brain_copy_2_future(self,hidd_layer,hidd_act_fn,output_act_fn,mean,stddev):
        if self.is_actor_brain_present:            
            self.weights_init_critic_copy_2(hidd_layer,mean,stddev)        
            num_hidden_layers = len(hidd_layer)          
            if hidd_act_fn == "relu":      
                layer_h = tf.nn.relu(tf.add(tf.matmul(self.state_in_future, self.weights_critic_copy_2['h_0']), self.biases_critic_copy_2['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.relu(tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format(i+1)]), self.biases_critic_copy_2['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format("out")]), self.biases_critic_copy_2['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format("out")]), self.biases_critic_copy_2['b_{0}'.format("out")]))                      
                self.output_critic_copy_2 = tf.reshape(layer_out_critic,[-1])
                self.output_critic_copy_2 = tf.stop_gradient(self.output_critic_copy_2)
                self.critic_tvar_copy_2_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_copy_2_present = True  
            elif hidd_act_fn == "sigmoid":      
                layer_h = tf.nn.sigmoid(tf.add(tf.matmul(self.state_in_future, self.weights_critic_copy_2['h_0']), self.biases_critic_copy_2['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.sigmoid(tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format(i+1)]), self.biases_critic_copy_2['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format("out")]), self.biases_critic_copy_2['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic_copy_2['h_{0}'.format("out")]), self.biases_critic_copy_2['b_{0}'.format("out")]))                      
                self.output_critic_copy_2 = tf.reshape(layer_out_critic,[-1])
                self.output_critic_copy_2 = tf.stop_gradient(self.output_critic_copy_2)
                self.critic_tvar_copy_2_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_copy_2_present = True                  
                
        else:
            print("please create actor brain first")
     
    def create_actor_brain(self,hidd_layer,hidd_act_fn,output_act_fn,mean,stddev):        
        self.is_actor_brain_present =  True
        self.weights_init_actor(hidd_layer,mean,stddev)        
        num_hidden_layers = len(hidd_layer)   
        
        if hidd_act_fn == "sigmoid":
            layer_h = tf.nn.sigmoid(tf.add(tf.matmul(self.state_in, self.weights_actor['h_0']), self.biases_actor['b_0']))
            for i in range(num_hidden_layers):
                if i < num_hidden_layers-1:          
                    layer_h = tf.nn.sigmoid(tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format(i+1)]), self.biases_actor['b_{0}'.format(i+1)]))
                else:
                    if self.is_action_continous:
                        layer_out = tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format("out")]), self.biases_actor['b_{0}'.format("out")])
                    elif self.is_action_discrete:
                        layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format("out")]), self.biases_actor['b_{0}'.format("out")]))   
        elif hidd_act_fn == "relu":
            layer_h = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights_actor['h_0']), self.biases_actor['b_0']))
            for i in range(num_hidden_layers):
                if i < num_hidden_layers-1:          
                    layer_h = tf.nn.relu(tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format(i+1)]), self.biases_actor['b_{0}'.format(i+1)]))
                else:
                    if self.is_action_continous:
                        layer_out = tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format("out")]), self.biases_actor['b_{0}'.format("out")])
                    elif self.is_action_discrete:
                        layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_h, self.weights_actor['h_{0}'.format("out")]), self.biases_actor['b_{0}'.format("out")])) 
        else:
            raise Exception("only sigmoid and relu values are supported")
        
        if self.is_action_continous:
            self.output_actor_mean = tf.nn.softmax(layer_out[:,:self.a_size]) #choose wether is one action or the other thats why softmax, cannot be both
            self.output_actor_std = tf.nn.sigmoid(layer_out[:,self.a_size:])/10.0       
            self.actor_tvar_num = (num_hidden_layers+1)*2
                       
        elif self.is_action_discrete:
            self.action_dist=layer_out
            self.actor_tvar_num = (num_hidden_layers+1)*2
            
    def create_critic_brain(self,hidd_layer,hidd_act_fn,output_act_fn,mean,stddev):
        if self.is_actor_brain_present:            
            self.weights_init_critic(hidd_layer,mean,stddev)        
            num_hidden_layers = len(hidd_layer)          
            if hidd_act_fn == "relu":      
                layer_h = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights_critic['h_0']), self.biases_critic['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.relu(tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format(i+1)]), self.biases_critic['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format("out")]), self.biases_critic['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format("out")]), self.biases_critic['b_{0}'.format("out")]))                      
                self.output_critic = tf.reshape(layer_out_critic,[-1])
                self.critic_tvar_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_present = True
            elif hidd_act_fn == "sigmoid":      
                layer_h = tf.nn.sigmoid(tf.add(tf.matmul(self.state_in, self.weights_critic['h_0']), self.biases_critic['b_0']))
                for i in range(num_hidden_layers):
                    if i < num_hidden_layers-1:          
                        layer_h = tf.nn.sigmoid(tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format(i+1)]), self.biases_critic['b_{0}'.format(i+1)]))
                    else:
                        if output_act_fn == "linear":
                            layer_out_critic = tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format("out")]), self.biases_critic['b_{0}'.format("out")])
                        elif output_act_fn == "tanh":
                            layer_out_critic = tf.nn.tanh(tf.add(tf.matmul(layer_h, self.weights_critic['h_{0}'.format("out")]), self.biases_critic['b_{0}'.format("out")]))                      
                self.output_critic = tf.reshape(layer_out_critic,[-1])
                self.critic_tvar_num = (num_hidden_layers+1)*2   
                self.is_critic_brain_present = True                  
        else:
            print("please create actor brain first")
        self.create_critic_brain_copy_1(hidd_layer,hidd_act_fn,output_act_fn,mean,stddev)
        self.create_critic_brain_copy_2_future(hidd_layer,hidd_act_fn,output_act_fn,mean,stddev)

    def calculate_delta(self):    
        self.new_delta = tf.add(self.reward_placeholder,tf.add(tf.cast(tf.math.logical_not(self.d_placeholder),tf.float32)*self.gamma_placeholder*self.output_critic_copy_2, -self.output_critic_copy_1))
        
    def critic_loss_function(self):
        self.critic = -self.new_delta*self.output_critic      
        
    def actor_loss_function(self):
        self.actor = -self.I_placeholder*self.new_delta*tf.log(self.responsible_outputs)
    
    def calculate_new_actor_loss_gradient(self):
        self.actor_gradients = tf.gradients(self.actor,self.tvars[:self.actor_tvar_num],name="gradientone")
   
    def calculate_new_critic_gradients(self):       
        self.critic_gradients = tf.gradients(self.critic,self.tvars[self.actor_tvar_num:self.actor_tvar_num+self.critic_tvar_num],name="gradienttwo")
  
    def new_update_actor_weights(self):
        self.new_new_update_actor_weights_op = self.optimizer_actor.apply_gradients(grads_and_vars=zip(self.actor_gradients,self.tvars[:self.actor_tvar_num]),global_step=self.global_step_actor)#,global_step=self.global_step_actor) 
            

    def new_update_critic_weights(self):
        self.new_update_critic_weights_op = self.optimizer_critic.apply_gradients(grads_and_vars=zip(self.critic_gradients,self.tvars[self.actor_tvar_num:self.actor_tvar_num+self.critic_tvar_num]),global_step=self.global_step_critic)#,global_step=self.global_step_critic)   

    def update_critic_copies_weigths(self):
        self.update_critic_copy_weights_1 = list(np.zeros(self.critic_tvar_num))
        self.update_critic_copy_weights_2 = list(np.zeros(self.critic_tvar_num)) 
        
        for i in range(self.critic_tvar_num):
            self.update_critic_copy_weights_1[i] =  tf.assign(self.tvars[self.actor_tvar_num+self.critic_tvar_num+i],self.tvars[self.actor_tvar_num+i])
            self.update_critic_copy_weights_2[i] =  tf.assign(self.tvars[self.actor_tvar_num+2*self.critic_tvar_num+i],self.tvars[self.actor_tvar_num+i]) 
            
    def set_data_indexes(self):

        if self.is_action_discrete:        
            self.index_a = [0,self.cont_offset]
            self.index_I = [1+self.cont_offset,1+self.cont_offset]
            self.index_gamma = [2+self.cont_offset,2+self.cont_offset]
            self.index_s0 = [3+self.cont_offset,3+self.cont_offset+self.s_size]
            self.index_s1 = [3+self.cont_offset+self.s_size,3+self.cont_offset+2*self.s_size]
            self.index_r = [3+self.cont_offset+2*self.s_size,3+self.cont_offset+2*self.s_size]
            self.index_d = [3+self.cont_offset+2*self.s_size+1,3+self.cont_offset+2*self.s_size+1]
        else:
            self.index_a = [0,self.cont_offset]
            self.index_I = [self.cont_offset,self.cont_offset]
            self.index_gamma = [1+self.cont_offset,1+self.cont_offset]
            self.index_s0 = [1+self.cont_offset,1+self.cont_offset+self.s_size]
            self.index_s1 = [2+self.cont_offset+self.s_size,2+self.cont_offset+2*self.s_size]
            self.index_r = [2+self.cont_offset+2*self.s_size,2+self.cont_offset+2*self.s_size]
            self.index_d = [2+self.cont_offset+2*self.s_size+1,2+self.cont_offset+2*self.s_size+1]          
        
    def memorize_data(self,a,s0,s1,r,d):        
        if self.is_action_discrete:
            self.memories_holder[self.internal_memory_counter,0] = a  #a   
            self.memories_holder[self.internal_memory_counter,1+self.cont_offset] = self.I*(self.gamma**self.internal_step_counter)  #I       
            self.memories_holder[self.internal_memory_counter,2+self.cont_offset] = self.gamma #gamma               
            self.memories_holder[self.internal_memory_counter,3+self.cont_offset:3+self.cont_offset+self.s_size] = s0  #s0        
            self.memories_holder[self.internal_memory_counter,3+self.cont_offset+self.s_size:3+self.cont_offset+2*self.s_size] = s1     #s1           
            self.memories_holder[self.internal_memory_counter,3+self.cont_offset+2*self.s_size] = r  #r        
            self.memories_holder[self.internal_memory_counter,3+self.cont_offset+2*self.s_size+1] = d  #d 
            
            
            
            
        else:
            self.memories_holder[self.internal_memory_counter,0:self.cont_offset] = a  #a              
            self.memories_holder[self.internal_memory_counter,self.cont_offset] = self.I*(self.gamma**self.internal_step_counter)  #I       
            self.memories_holder[self.internal_memory_counter,1+self.cont_offset] = self.gamma #gamma               
            self.memories_holder[self.internal_memory_counter,2+self.cont_offset:2+self.cont_offset+self.s_size] = s0  #s0        
            self.memories_holder[self.internal_memory_counter,2+self.cont_offset+self.s_size:2+self.cont_offset+2*self.s_size] = s1     #s1           
            self.memories_holder[self.internal_memory_counter,2+self.cont_offset+2*self.s_size] = r  #r        
            self.memories_holder[self.internal_memory_counter,2+self.cont_offset+2*self.s_size+1] = d  #d   
               
     
        if d:
            self.internal_step_counter = 0 
        else:
            self.internal_step_counter = self.internal_step_counter +1
        
        self.internal_memory_counter = self.internal_memory_counter + 1
        
        if self.internal_memory_counter >= self.amount_of_data_to_memorize:
            return True
        else:
            return False
 
    def reset_memories(self):
         self.memories_holder = np.zeros([self.amount_of_data_to_memorize,self.size_of_data])        
         self.internal_memory_counter = 0
 
    def shuffle_memories(self):
        np.random.shuffle(self.memories_holder)
    
    def get_memories(self):
        return self.memories_holder

    def set_critic_learning_rate_decay(self,optimizer="GRAD",type_of_decay="exponential",learning_rate = 0.001,decay_steps = 5000, decay_rate = 0.5, end_learning_rate = 0.000001):
        
        if type_of_decay == "exponential":
            self.learning_rate_critic= learning_rate
            self.decay_steps_critic= decay_steps
            self.decay_rate_critic= decay_rate        # must be <1
            self.global_step_critic = tf.Variable(0, trainable=False)
            self.learning_rate_critic = tf.train.exponential_decay(
                                                               self.learning_rate_critic,
                                                               self.global_step_critic,
                                                               self.decay_steps_critic,
                                                               self.decay_rate_critic,
                                                               staircase=False,
                                                               name=None
                                                                        )        
            
        elif type_of_decay == "polinomial":
            
            self.global_step_critic = tf.Variable(0, trainable=False)
            self.starter_learning_rate_critic = learning_rate
            self.end_learning_rate_critic = end_learning_rate
            self.decay_steps_critic = decay_steps
            self.learning_rate_critic = tf.train.polynomial_decay(self.starter_learning_rate_critic, self.global_step_critic,
                                              self.decay_steps_critic, self.end_learning_rate_critic,
                                              power=decay_rate) 

        elif type_of_decay == "none":
            self.learning_rate_critic = learning_rate
            self.global_step_critic = tf.Variable(0, trainable=False)
            
        elif  type_of_decay != "exponential" and type_of_decay != "polinomial" and type_of_decay != "none":
            raise Exception("Only exponential, polinomial, and none decay types are supported")
        
        if optimizer == "ADAM":
            self.optimizer_critic = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
            
        elif optimizer == "GRAD":
            self.optimizer_critic = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_critic)
        
        self.is_critic_decay_set = True
        
        
    def set_actor_learning_rate_decay(self,optimizer="GRAD",type_of_decay="exponential",learning_rate = 0.001,decay_steps = 5000, decay_rate = 0.5, end_learning_rate = 0.000001):
        
        if type_of_decay == "exponential":
            self.learning_rate_actor= learning_rate
            self.decay_steps_actor= decay_steps
            self.decay_rate_actor= decay_rate        # must be <1
            self.global_step_actor = tf.Variable(0, trainable=False)
            self.learning_rate_actor = tf.train.exponential_decay(
                                                               self.learning_rate_actor,
                                                               self.global_step_actor,
                                                               self.decay_steps_actor,
                                                               self.decay_rate_actor,
                                                               staircase=False,
                                                               name=None
                                                                        )        
            
        elif type_of_decay == "polinomial":
            
            self.global_step_actor = tf.Variable(0, trainable=False)
            self.starter_learning_rate_actor = learning_rate
            self.end_learning_rate_actor = end_learning_rate
            self.decay_steps_actor = decay_steps
            self.learning_rate_actor = tf.train.polynomial_decay(self.starter_learning_rate_actor, self.global_step_actor,
                                              self.decay_steps_actor, self.end_learning_rate_actor,
                                              power=decay_rate) 

        elif type_of_decay == "none":
            self.learning_rate_actor = learning_rate
            self.global_step_actor = tf.Variable(0, trainable=False)
            
        elif  type_of_decay != "exponential" and type_of_decay != "polinomial" and type_of_decay != "none":
            raise Exception("Only exponential and polinomial decay types are supported")
        
        if optimizer == "ADAM":
            self.optimizer_actor = tf.train.AdamOptimizer(learning_rate=self.learning_rate_actor)
            
        elif optimizer == "GRAD":
            self.optimizer_actor = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_actor) 

        self.is_actor_decay_set = True    

    def create_new_graph_connections(self):
        if self.is_actor_brain_present and self.is_critic_brain_present and self.is_actor_decay_set and self.is_critic_decay_set:
            
            self.calculate_delta()
            
            self.normal_dist_prob()    
            
            self.actor_loss_function()
            
            self.critic_loss_function()
            
            self.tvars = tf.trainable_variables()
            
            self.calculate_new_actor_loss_gradient()
            
            self.calculate_new_critic_gradients()
            
            self.new_update_actor_weights()
            
            self.new_update_critic_weights()
            
            self.update_critic_copies_weigths()
                       
            self.saver = tf.train.Saver()
        else:
            raise Exception("initialize actor and critic brains first then set the decay of the learning rates")
        
        self.init = tf.global_variables_initializer()
  
    def normal_dist_prob(self): 
        
        if self.is_action_continous:
            y = tf.reduce_sum(tf.square((self.action_placeholder-self.output_actor_mean))*self.output_actor_std**(-1),axis=1) #GRAPH AGAIN?? THE GRAPH DOESNT CHANGE!
            Z = (2*np.pi)**(0.5*self.a_size)*(tf.reduce_prod(self.output_actor_std,axis=1))**(0.5)  
            self.pdf = tf.exp(-0.5*y)/Z
            self.responsible_outputs =  self.pdf
        elif self.is_action_discrete:
            self.pdf = self.action_dist
            self.indexes = tf.range(0, tf.shape(self.action_dist)[0]) * tf.shape(self.action_dist)[1] + self.action_placeholder        
            self.responsible_outputs = tf.gather(tf.reshape(self.action_dist, [-1]), self.indexes)    

    def sample_action(self,sess,state): #CHECK
        if self.is_action_continous:
            try:
                tvars,mean,cov= sess.run([self.tvars,self.output_actor_mean,self.output_actor_std],feed_dict={self.state_in:state})
            
                
                cov = np.diag(cov[0])
                sample_holder = np.random.multivariate_normal(mean[0],cov)
            except:
                print(tvars)
                return -1
        elif self.is_action_discrete:
            
            self.action_dist_sample = sess.run([self.action_dist],feed_dict={self.state_in:state})
            if (np.nan in self.action_dist_sample[0][0]) or (None in self.action_dist_sample[0][0]):
                print(self.action_dist_sample)           
            sample_holder = np.zeros([self.action_dist_sample[0].shape[0],1])
            actions_pool = list(range(self.a_size))
            
            for i in range(self.action_dist_sample[0].shape[0]):
                sample_holder[i] = np.random.choice(actions_pool,p=self.action_dist_sample[0][i,:])      
        return sample_holder

###################################END OF CLASS DECLARATION ######################################################################
##################################################################################################################################        

##### ENVIRONMENT CREATION #########  
    
#env = gym.make('BipedalWalker-v2')
env = gym.make('CartPole-v0')

env.seed(0)
#uper_action_limit = env.action_space.high
#lower_action_limit = env.action_space.low
uper_action_limit = np.array([10,10])
lower_action_limit = -np.array([10,10])

#------------loops parameters------------------------------
num_mem = 300 #amount of samples collected before training
ratio = 3  #we devide the shuffle memories in ratio number of sets whose size is "mini_batch_size"
mini_batch_size = num_mem/ratio ##mini_batch_size = num_mem / ratio must be integer
mini_batch_size = int(mini_batch_size)
#----------------------------------------------------------

#-------------plot arrays initialization-------------------
num_episodes = 26000 # episodes
plotlist = list(np.zeros(num_episodes)) #plot array
plot_list_avg = list(np.zeros(int(num_episodes/100.0)))
#----------------------------------------------------------


#-------------Agent instantiation -------------------------
tf.reset_default_graph()
agent= agent_episodic_continuous_action(gamma=1,s_size=4,a_size=2,action_type="discrete",amount_of_data_to_memorize=num_mem,optimizer="GRAD")

agent.create_actor_brain(hidd_layer=[8],hidd_act_fn="relu",output_act_fn="linear",mean=0.0,stddev=0.01)
agent.create_critic_brain(hidd_layer=[8],hidd_act_fn="relu",output_act_fn="linear",mean=0.0,stddev=0.01)
agent.set_actor_learning_rate_decay(optimizer="GRAD",type_of_decay="exponential",learning_rate = 0.001,decay_steps = 5000, decay_rate = 0.1)
agent.set_critic_learning_rate_decay(optimizer="GRAD",type_of_decay="exponential",learning_rate = 0.001,decay_steps = 5000, decay_rate = 0.1)

agent.create_new_graph_connections()
#----------------------------------------------------------

#--------------Counters------------------------------------
plot_count=0
count=1
count_avg = 0
avg_counter = 0
#----------------------------------------------------------

#-------------Save model ----------------------------------
are_we_saving = False
path_to_save = "some/path/to/somewhere"
#----------------------------------------------------------
with tf.Session() as sess:
    sess.run(agent.init)
    sess.graph.finalize()
    
    for i in range(num_episodes):
        d = False
        s0 = env.reset()
        plotlist[plot_count]=count
        plot_count =plot_count +1
        count_avg = count_avg + count
        count = 0
        if print_each_every(i,100,"Episode: {0} Average: {1}".format(i,count_avg/100.0)):
            plot_list_avg[avg_counter] = count_avg/100.0  #we gather the avgs in order to plot them later  
            avg_counter = avg_counter + 1 # we move to the next cell
            count_avg = 0 #we start the avg all over again
        while not d:

            if agent.is_action_continous:            
                avect = agent.sample_action(sess,[s0])            
                la = np.random.choice([(avect[0]>0.5)*1,1-(avect[1]>0.5)*1])
                a = int(la)
                s1,r,d,_ = env.step(int(la))
                full_memory=agent.memorize_data(a=avect,s0=s0,s1=s1,r=r,d=d)
            else:          
                a = agent.sample_action(sess,[s0])[0][0]
              #  a = np.random.randint(2)
                a = int(a)
                s1,r,d,_ = env.step(a)                        
                full_memory=agent.memorize_data(a=a,s0=s0,s1=s1,r=r,d=d)
            
            s0 = s1
            print_each_every(count+1,200,"Reached step: {0}".format(200))
            if full_memory:
                
                agent.shuffle_memories()
                batch = agent.get_memories()
                agent.reset_memories() 
                for mini_batch_step in range(ratio):
                    sess.run([agent.update_critic_copy_weights_1,agent.update_critic_copy_weights_2]) #Update copies weights
                    
                    
                    start =  mini_batch_step*mini_batch_size
                    stop = (mini_batch_step+1)*mini_batch_size
                    
                    if agent.is_action_continous:
                        #FOR DEBUG
#                       raw_critic_1,raw_critic_2,raw_delta,raw_critic,raw_actor,raw_tvars= sess.run([agent.output_critic_copy_1,agent.output_critic_copy_2,agent.new_delta,agent.output_critic,agent.pdf,agent.tvars],feed_dict = {agent.state_in:batch[start:stop,agent.index_s0[0]:agent.index_s0[1]]\
#                                                                                                                     ,agent.state_in_future:batch[start:stop,agent.index_s1[0]:agent.index_s1[1]]\
#                                                                                                                      ,agent.reward_placeholder:batch[start:stop,agent.index_r[0]]\
#                                                                                                                      ,agent.action_placeholder:batch[start:stop,agent.index_a[0]:agent.index_a[1]]\
#                                                                                                                      ,agent.I_placeholder:batch[start:stop,agent.index_I[0]]\
#                                                                                                                      ,agent.d_placeholder:batch[start:stop,agent.index_d[0]]\
#                                                                                                                      ,agent.gamma_placeholder:batch[start:stop,agent.index_gamma[0]]\
#                                                                                                                      })
                        
                        
                        
                #       raise Exception("1 test") 
                        
                       delta,_,_= sess.run([agent.new_delta,agent.new_new_update_actor_weights_op,agent.new_update_critic_weights_op],feed_dict = {agent.state_in:batch[start:stop,agent.index_s0[0]:agent.index_s0[1]]\
                                                                                                                     ,agent.state_in_future:batch[start:stop,agent.index_s1[0]:agent.index_s1[1]]\
                                                                                                                      ,agent.reward_placeholder:batch[start:stop,agent.index_r[0]]\
                                                                                                                      ,agent.action_placeholder:batch[start:stop,agent.index_a[0]:agent.index_a[1]]\
                                                                                                                      ,agent.I_placeholder:batch[start:stop,agent.index_I[0]]\
                                                                                                                      ,agent.d_placeholder:batch[start:stop,agent.index_d[0]]\
                                                                                                                      ,agent.gamma_placeholder:batch[start:stop,agent.index_gamma[0]]\
                                                                                                                      })
#                       print(delta[0]) 
                    else:
                        sess.run([agent.new_new_update_actor_weights_op,agent.new_update_critic_weights_op],feed_dict = {agent.state_in:batch[start:stop,agent.index_s0[0]:agent.index_s0[1]]\
                                                                                                                      ,agent.state_in_future:batch[start:stop,agent.index_s1[0]:agent.index_s1[1]]\
                                                                                                                      ,agent.reward_placeholder:batch[start:stop,agent.index_r[0]]\
                                                                                                                      ,agent.action_placeholder:batch[start:stop,agent.index_a[0]]\
                                                                                                                      ,agent.I_placeholder:batch[start:stop,agent.index_I[0]]\
                                                                                                                      ,agent.d_placeholder:batch[start:stop,agent.index_d[0]]\
                                                                                                                      ,agent.gamma_placeholder:batch[start:stop,agent.index_gamma[0]]\
                                                                                                                      })                        
                        
                        
                        
                        
                        
                        
    
    
    
                    sess.run([agent.update_critic_copy_weights_1,agent.update_critic_copy_weights_2]) #Update copies weights
            count = count +1
 
    if are_we_saving:           
        agent.save_model(path_to_save,sess)            
            


plt.plot(plotlist,'bx')
plt.ylabel("reward")
plt.show()

plt.plot(plot_list_avg,'kx')
plt.ylabel("avg reward per 100 episodes")
plt.show()    

    




