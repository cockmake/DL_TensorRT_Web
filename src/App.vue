<script setup>
import {ref} from "vue";
import axios from "axios"
// let video_url = axios.defaults.baseURL + '/ttt'
let img_url = ref('')
let additional_info = ref('')
let source = new EventSource(axios.defaults.baseURL + '/get_data');
source.onmessage = function(res) {
  let data_json = JSON.parse(res.data)
  img_url.value = 'data:image/jpg;base64,'+ data_json['img_base64']
  additional_info.value = data_json['boxes'] + '\n' + data_json['id']
}
</script>

<template>
  <div style="height: 500px; width: 500px;">
    <img alt="这里显示图像" :src="img_url" >
<!--    <img alt="这里显示图像" :src="video_url">-->
    <div style="white-space: pre-wrap;">
      {{additional_info}}
    </div>
  </div>
</template>

<style scoped>
</style>
