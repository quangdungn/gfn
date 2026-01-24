
# Create checkpoint directory
mkdir -p ./checkpoints

echo ""
echo "================================================================"
echo "SENTIMENT CLASSIFICATION"
echo "================================================================"

python gfn_vietnamese_pipeline.py \
    --task sentiment \
    --data_dir ./data \
    --save_dir ./checkpoints/sentiment \
    --window_size 20 \
    --min_freq 1 \
    --embedding_dim 300 \
    --hidden_dim 300 \
    --num_heads 3 \
    --dropout 0.5 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --fusion_lr 0.05 \
    --stage1_epochs 100 \
    --stage1_patience 10 \
    --stage2_iterations 1000 \
    --stage2_patience 100

if [ $? -ne 0 ]; then
    echo "Sentiment training failed!"
    exit 1
fi

echo ""
echo "================================================================"
echo "TOPIC CLASSIFICATION"
echo "================================================================"

python gfn_vietnamese_pipeline.py \
    --task topic \
    --data_dir ./data \
    --save_dir ./checkpoints/topic \
    --window_size 20 \
    --min_freq 1 \
    --embedding_dim 300 \
    --hidden_dim 300 \
    --num_heads 3 \
    --dropout 0.5 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --fusion_lr 0.05 \
    --stage1_epochs 100 \
    --stage1_patience 10 \
    --stage2_iterations 1000 \
    --stage2_patience 100

if [ $? -ne 0 ]; then
    echo "Topic training failed!"
    exit 1
fi

echo ""
echo "================================================================"
echo "TRAINING SUMMARY"
echo "================================================================"

if [ -f "./checkpoints/sentiment/results.json" ]; then
    echo ""
    echo "Sentiment Classification Results:"
    cat ./checkpoints/sentiment/results.json | grep -E "test_accuracy|test_micro_f1|test_macro_f1"
fi

if [ -f "./checkpoints/topic/results.json" ]; then
    echo ""
    echo "Topic Classification Results:"
    cat ./checkpoints/topic/results.json | grep -E "test_accuracy|test_micro_f1|test_macro_f1"
fi

echo ""
echo "================================================================"
echo "Training Complete! Results saved in ./checkpoints/"
echo "================================================================"